#!/usr/bin/env python3
"""Convert the HINYEUN LeRobot RGB-D dataset to a right-arm DP3 Zarr dataset.

The source dataset is never modified. The converter keeps only the right arm
state/action fields, reconstructs gravity-aligned XYZ point clouds from Orbbec
depth, applies a fixed workspace crop, and writes fixed-size point clouds.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from visualize_lerobot_pointcloud import (
    ORBBEC_DEPTH_KEY,
    align_with_gravity,
    crop_point_cloud,
    load_gravity_direction,
    reconstruct_point_cloud,
    to_numpy,
)


STATE_KEY = "observation.state"
ACTION_KEY = "action"

RIGHT_STATE_SLICE = slice(8, 16)
RIGHT_ACTION_SLICE = slice(8, 17)

DEFAULT_CROP_MIN = (-0.222791, 0.238951, -0.282919)
DEFAULT_CROP_MAX = (0.469885, 0.774905, 0.075385)

DEPTH_VIDEO_CHUNK_KEY = f"videos/{ORBBEC_DEPTH_KEY}/chunk_index"
DEPTH_VIDEO_FILE_KEY = f"videos/{ORBBEC_DEPTH_KEY}/file_index"
DEPTH_VIDEO_FROM_TIMESTAMP_KEY = f"videos/{ORBBEC_DEPTH_KEY}/from_timestamp"
DEPTH_VIDEO_TO_TIMESTAMP_KEY = f"videos/{ORBBEC_DEPTH_KEY}/to_timestamp"


class SequentialDepthVideoReader:
    """Decode consolidated depth videos once and query increasing timestamps.

    LeRobot's PyAV random-access decoder can land on the next GOP keyframe when
    seeking immediately before that keyframe. Conversion visits source frames
    in time order, so sequential decoding is both exact and substantially
    faster than opening and seeking the video once per output frame.
    """

    def __init__(self, depth_info: dict, max_timestamp_error_s: float):
        self.depth_info = depth_info
        self.max_timestamp_error_s = max_timestamp_error_s
        self.path: Path | None = None
        self.container = None
        self.stream = None
        self.frames = None
        self.previous = None
        self.current = None
        self.last_query_timestamp = -np.inf
        self.max_observed_error_s = 0.0

    def close(self) -> None:
        if self.container is not None:
            self.container.close()
        self.path = None
        self.container = None
        self.stream = None
        self.frames = None
        self.previous = None
        self.current = None
        self.last_query_timestamp = -np.inf

    def open(self, path: Path) -> None:
        import av

        resolved_path = path.resolve()
        if self.path == resolved_path:
            return
        self.close()
        self.path = resolved_path
        self.container = av.open(str(resolved_path))
        self.stream = self.container.streams.video[0]
        self.frames = iter(self.container.decode(self.stream))
        self.current = self._read_next()

    def _read_next(self):
        if self.frames is None or self.stream is None:
            return None
        for frame in self.frames:
            if frame.pts is not None:
                return float(frame.pts * self.stream.time_base), frame
        return None

    def read_at(self, timestamp: float) -> np.ndarray:
        from lerobot.datasets.depth_utils import dequantize_depth

        if self.path is None:
            raise RuntimeError("Open a depth video before reading it")
        if timestamp + 1e-7 < self.last_query_timestamp:
            raise ValueError(
                f"Depth timestamps must be increasing within one video: "
                f"{timestamp} follows {self.last_query_timestamp}"
            )
        self.last_query_timestamp = timestamp

        while self.current is not None and self.current[0] + 1e-7 < timestamp:
            self.previous = self.current
            self.current = self._read_next()

        candidates = [item for item in (self.previous, self.current) if item is not None]
        if not candidates:
            raise EOFError(f"No frames decoded from depth video {self.path}")
        selected_timestamp, selected_frame = min(
            candidates, key=lambda item: abs(item[0] - timestamp)
        )
        timestamp_error = abs(selected_timestamp - timestamp)
        self.max_observed_error_s = max(self.max_observed_error_s, timestamp_error)
        if timestamp_error > self.max_timestamp_error_s:
            raise ValueError(
                f"Depth/video timestamp mismatch in {self.path}: requested "
                f"{timestamp:.6f}s, nearest frame {selected_timestamp:.6f}s, "
                f"error {timestamp_error:.6f}s"
            )

        quantized = selected_frame.to_ndarray(format="gray12le")
        return dequantize_depth(
            quantized,
            depth_min=float(self.depth_info.get("video.depth_min", 0.0)),
            depth_max=float(self.depth_info.get("video.depth_max", 5.0)),
            shift=float(self.depth_info.get("video.shift", 0.0)),
            use_log=bool(self.depth_info.get("video.use_log", False)),
            output_unit="mm",
            output_tensor=False,
            output_channel_last=False,
        )


def make_depth_only_dataset_view(dataset_root: Path):
    """Create a temporary LeRobot view containing only the required video key.

    It also hides macOS AppleDouble files. Ordinary files are symlinked, so the
    4 GB source dataset is not copied. ``meta/info.json`` is rewritten in the
    temporary directory to prevent LeRobot from decoding unused RGB streams.
    """
    temporary_dir = tempfile.TemporaryDirectory(prefix="lerobot-dp3-")
    clean_root = Path(temporary_dir.name)

    for source in dataset_root.rglob("*"):
        if source.name.startswith("._") or source.name == ".DS_Store":
            continue

        relative = source.relative_to(dataset_root)
        parts = relative.parts
        if parts and parts[0] == "videos":
            if len(parts) >= 2 and parts[1] != ORBBEC_DEPTH_KEY:
                continue

        destination = clean_root / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue

        if not source.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative.as_posix() == "meta/info.json":
            info = json.loads(source.read_text(encoding="utf-8"))
            info["features"] = {
                key: value
                for key, value in info["features"].items()
                if value.get("dtype") != "video" or key == ORBBEC_DEPTH_KEY
            }
            destination.write_text(json.dumps(info, indent=4), encoding="utf-8")
        else:
            destination.symlink_to(source.resolve())

    return clean_root, temporary_dir


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0 or len(points) == 0:
        return points
    voxel_coordinates = np.floor(points / voxel_size).astype(np.int32)
    _, first_indices = np.unique(voxel_coordinates, axis=0, return_index=True)
    return points[np.sort(first_indices)]


def farthest_point_sample(points: np.ndarray, num_points: int) -> np.ndarray:
    """Deterministic NumPy farthest-point sampling over XYZ points."""
    n_points = len(points)
    if n_points == 0:
        raise ValueError("Cannot sample an empty point cloud")
    if n_points <= num_points:
        repeated_indices = np.resize(np.arange(n_points), num_points)
        return points[repeated_indices]

    sampled_indices = np.empty(num_points, dtype=np.int64)
    min_squared_distance = np.full(n_points, np.inf, dtype=np.float32)
    center = points.mean(axis=0)
    farthest = int(np.argmax(np.sum((points - center) ** 2, axis=1)))

    for sample_index in range(num_points):
        sampled_indices[sample_index] = farthest
        delta = points - points[farthest]
        squared_distance = np.einsum("ij,ij->i", delta, delta)
        np.minimum(min_squared_distance, squared_distance, out=min_squared_distance)
        farthest = int(np.argmax(min_squared_distance))
    return points[sampled_indices]


def fixed_size_sample(
    points: np.ndarray,
    num_points: int,
    method: str,
    voxel_size: float,
    seed: int,
) -> np.ndarray:
    candidates = voxel_downsample(points, voxel_size)
    if len(candidates) < num_points and len(points) >= num_points:
        candidates = points

    if method == "fps":
        sampled = farthest_point_sample(candidates, num_points)
    elif method == "uniform":
        if len(candidates) < num_points:
            indices = np.resize(np.arange(len(candidates)), num_points)
        else:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(candidates), size=num_points, replace=False)
        sampled = candidates[indices]
    else:
        raise ValueError(f"Unsupported sampling method: {method}")
    return sampled.astype(np.float32, copy=False)


def read_episode_metadata(dataset_root: Path, max_episodes: int | None):
    import pyarrow.parquet as pq

    episode_path = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    columns = [
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        DEPTH_VIDEO_CHUNK_KEY,
        DEPTH_VIDEO_FILE_KEY,
        DEPTH_VIDEO_FROM_TIMESTAMP_KEY,
        DEPTH_VIDEO_TO_TIMESTAMP_KEY,
    ]
    rows = pq.read_table(episode_path, columns=columns).to_pylist()
    rows.sort(key=lambda row: row["episode_index"])
    if max_episodes is not None:
        rows = rows[:max_episodes]
    if not rows:
        raise ValueError("No episodes selected for conversion")
    return rows


def selected_source_indices(episode_rows: list[dict], stride: int):
    selected_by_episode = []
    for row in episode_rows:
        start = int(row["dataset_from_index"])
        stop = int(row["dataset_to_index"])
        length = int(row["length"])
        if stop - start != length:
            raise ValueError(
                f"Episode {row['episode_index']} has inconsistent bounds: "
                f"length={length}, range=[{start}, {stop})"
            )
        selected_by_episode.append(np.arange(start, stop, stride, dtype=np.int64))
    return selected_by_episode


def prepare_output(
    output_path: Path,
    total_frames: int,
    num_episodes: int,
    num_points: int,
    chunk_length: int,
    overwrite: bool,
):
    import zarr
    from numcodecs import Blosc

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --overwrite to replace it."
            )
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(output_path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    pc_chunk_length = min(chunk_length, total_frames)
    data.create_dataset(
        "point_cloud",
        shape=(total_frames, num_points, 3),
        chunks=(pc_chunk_length, num_points, 3),
        dtype="f4",
        compressor=compressor,
    )
    data.create_dataset(
        "state",
        shape=(total_frames, 8),
        chunks=(min(1024, total_frames), 8),
        dtype="f4",
        compressor=compressor,
    )
    data.create_dataset(
        "action",
        shape=(total_frames, 9),
        chunks=(min(1024, total_frames), 9),
        dtype="f4",
        compressor=compressor,
    )
    meta.create_dataset(
        "episode_ends",
        shape=(num_episodes,),
        chunks=(num_episodes,),
        dtype="i8",
        compressor=compressor,
    )
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/hinyeun_glue_0714_rgbd")
    parser.add_argument("--target-fps", type=int, default=10)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=1.8)
    parser.add_argument("--crop-min", type=float, nargs=3, default=DEFAULT_CROP_MIN)
    parser.add_argument("--crop-max", type=float, nargs=3, default=DEFAULT_CROP_MAX)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--sampling", choices=("fps", "uniform"), default="fps")
    parser.add_argument("--chunk-length", type=int, default=64)
    parser.add_argument("--max-video-timestamp-error-s", type=float, default=0.02)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.target_fps <= 0:
        parser.error("--target-fps must be positive")
    if args.num_points <= 0:
        parser.error("--num-points must be positive")
    if args.pixel_stride <= 0:
        parser.error("--pixel-stride must be positive")
    if args.max_video_timestamp_error_s <= 0:
        parser.error("--max-video-timestamp-error-s must be positive")
    crop_min = np.asarray(args.crop_min)
    crop_max = np.asarray(args.crop_max)
    if np.any(crop_min >= crop_max):
        parser.error("Every --crop-min value must be smaller than --crop-max")
    return args


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    source_fps = int(info["fps"])
    stride_float = source_fps / args.target_fps
    stride = int(round(stride_float))
    if stride < 1 or not np.isclose(stride_float, stride):
        raise ValueError(
            f"target-fps {args.target_fps} must evenly divide source fps {source_fps}"
        )

    episode_rows = read_episode_metadata(dataset_root, args.max_episodes)
    source_indices_by_episode = selected_source_indices(episode_rows, stride)
    total_frames = sum(len(indices) for indices in source_indices_by_episode)
    crop_min = np.asarray(args.crop_min, dtype=np.float32)
    crop_max = np.asarray(args.crop_max, dtype=np.float32)
    gravity_path = dataset_root / "meta" / "orbbec_gravity.yaml"
    gravity_down = load_gravity_direction(gravity_path)

    print(f"Source dataset: {dataset_root}")
    print(f"Output dataset: {output_path}")
    print(f"Episodes: {len(episode_rows)}")
    print(f"Frames: {total_frames} ({source_fps} Hz -> {args.target_fps} Hz, stride {stride})")
    print(f"Point cloud: {args.num_points} XYZ points, sampling={args.sampling}")
    print(f"Crop: min={crop_min.tolist()}, max={crop_max.tolist()}")

    root = prepare_output(
        output_path=output_path,
        total_frames=total_frames,
        num_episodes=len(episode_rows),
        num_points=args.num_points,
        chunk_length=args.chunk_length,
        overwrite=args.overwrite,
    )
    root.attrs.update(
        {
            "conversion_complete": False,
            "source_dataset": str(dataset_root),
            "source_repo_id": args.repo_id,
            "source_fps": source_fps,
            "target_fps": args.target_fps,
            "source_frame_stride": stride,
            "num_points": args.num_points,
            "pixel_stride": args.pixel_stride,
            "min_depth_m": args.min_depth_m,
            "max_depth_m": args.max_depth_m,
            "crop_min": crop_min.tolist(),
            "crop_max": crop_max.tolist(),
            "gravity_down_camera": gravity_down.tolist(),
            "gravity_aligned": True,
            "voxel_size": args.voxel_size,
            "sampling": args.sampling,
            "max_video_timestamp_error_s": args.max_video_timestamp_error_s,
            "state_source_slice": [8, 16],
            "action_source_slice": [8, 17],
            "state_names": info["features"][STATE_KEY]["names"][8:16],
            "action_names": info["features"][ACTION_KEY]["names"][8:17],
        }
    )

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "Install conversion dependencies with: "
            "pip install 'lerobot[dataset]==0.6.0' av==15.1.0 zarr==2.18.7"
        ) from exc

    loader_root, temporary_view = make_depth_only_dataset_view(dataset_root)
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=loader_root,
        video_backend="pyav",
        depth_output_unit="mm",
    )
    video_path_template = info["video_path"]
    depth_info = info["features"][ORBBEC_DEPTH_KEY].get("info", {})
    depth_reader = SequentialDepthVideoReader(
        depth_info=depth_info,
        max_timestamp_error_s=args.max_video_timestamp_error_s,
    )

    cursor = 0
    started_at = time.monotonic()
    with tqdm(total=total_frames, desc="Converting frames", unit="frame") as progress:
        for output_episode_index, (episode_row, source_indices) in enumerate(
            zip(episode_rows, source_indices_by_episode, strict=True)
        ):
            episode_points = np.empty(
                (len(source_indices), args.num_points, 3), dtype=np.float32
            )
            episode_states = np.empty((len(source_indices), 8), dtype=np.float32)
            episode_actions = np.empty((len(source_indices), 9), dtype=np.float32)

            depth_video_path = dataset_root / video_path_template.format(
                video_key=ORBBEC_DEPTH_KEY,
                chunk_index=int(episode_row[DEPTH_VIDEO_CHUNK_KEY]),
                file_index=int(episode_row[DEPTH_VIDEO_FILE_KEY]),
            )
            depth_reader.open(depth_video_path)
            episode_source_start = int(episode_row["dataset_from_index"])
            video_from_timestamp = float(episode_row[DEPTH_VIDEO_FROM_TIMESTAMP_KEY])

            for local_index, source_index in enumerate(source_indices):
                sample = dataset.get_raw_item(int(source_index))
                episode_frame_index = int(source_index) - episode_source_start
                video_timestamp = video_from_timestamp + episode_frame_index / source_fps
                sample[ORBBEC_DEPTH_KEY] = depth_reader.read_at(video_timestamp)
                point_cloud = reconstruct_point_cloud(
                    sample=sample,
                    pixel_stride=args.pixel_stride,
                    min_depth_m=args.min_depth_m,
                    max_depth_m=args.max_depth_m,
                    include_rgb=False,
                )
                point_cloud = align_with_gravity(point_cloud, gravity_down)
                point_cloud = crop_point_cloud(point_cloud, crop_min, crop_max)
                if len(point_cloud) == 0:
                    raise ValueError(
                        f"Source frame {source_index} has no points after cropping"
                    )
                point_cloud = fixed_size_sample(
                    points=point_cloud[:, :3],
                    num_points=args.num_points,
                    method=args.sampling,
                    voxel_size=args.voxel_size,
                    seed=int(source_index),
                )

                state = to_numpy(sample[STATE_KEY]).astype(np.float32, copy=False)
                action = to_numpy(sample[ACTION_KEY]).astype(np.float32, copy=False)
                episode_points[local_index] = point_cloud
                episode_states[local_index] = state[RIGHT_STATE_SLICE]
                episode_actions[local_index] = action[RIGHT_ACTION_SLICE]
                progress.update(1)

            next_cursor = cursor + len(source_indices)
            root["data/point_cloud"][cursor:next_cursor] = episode_points
            root["data/state"][cursor:next_cursor] = episode_states
            root["data/action"][cursor:next_cursor] = episode_actions
            root["meta/episode_ends"][output_episode_index] = next_cursor
            cursor = next_cursor
            root.attrs["frames_written"] = cursor

    depth_reader.close()
    del dataset
    del temporary_view
    root.attrs["conversion_complete"] = True
    root.attrs["frames_written"] = cursor
    root.attrs["elapsed_seconds"] = time.monotonic() - started_at
    root.attrs["max_observed_video_timestamp_error_s"] = depth_reader.max_observed_error_s

    print(f"Converted {cursor} frames into {output_path}")
    print(f"episode_ends: {root['meta/episode_ends'][:]}")
    for key in ("point_cloud", "state", "action"):
        array = root[f"data/{key}"]
        print(
            f"{key}: shape={array.shape}, dtype={array.dtype}, "
            f"range=[{float(array[:].min()):.6f}, {float(array[:].max()):.6f}]"
        )


if __name__ == "__main__":
    main()
