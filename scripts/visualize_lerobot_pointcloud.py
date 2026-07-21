#!/usr/bin/env python3
"""Reconstruct and inspect one RGB-D point cloud from a LeRobot dataset.

This is a preview/crop-bound tuning tool. It does not convert the complete
dataset to the fixed-size Zarr format expected by DP3 training.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np


ORBBEC_RGB_KEY = "observation.images.orbbec"
ORBBEC_DEPTH_KEY = "observation.images.orbbec_depth"
ORBBEC_INTRINSICS_KEY = "observation.camera.orbbec_intrinsics"
ORBBEC_DEPTH_SCALE_KEY = "observation.camera.orbbec_depth_scale_m"


def to_numpy(value) -> np.ndarray:
    """Convert a torch tensor or array-like value to a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def image_to_hwc(value: np.ndarray) -> np.ndarray:
    """Convert a CHW LeRobot image to HWC while accepting HWC as well."""
    image = to_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}")
    if image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    return image


def reconstruct_point_cloud(
    sample: dict,
    pixel_stride: int,
    min_depth_m: float,
    max_depth_m: float,
    include_rgb: bool,
) -> np.ndarray:
    """Back-project registered Orbbec depth into camera optical coordinates."""
    depth = image_to_hwc(sample[ORBBEC_DEPTH_KEY])[..., 0].astype(np.float32)
    intrinsics = to_numpy(sample[ORBBEC_INTRINSICS_KEY]).reshape(3, 3)
    depth_scale = float(to_numpy(sample[ORBBEC_DEPTH_SCALE_KEY]).reshape(-1)[0])

    rows = np.arange(0, depth.shape[0], pixel_stride)
    cols = np.arange(0, depth.shape[1], pixel_stride)
    u, v = np.meshgrid(cols, rows)
    depth_m = depth[::pixel_stride, ::pixel_stride] * depth_scale

    valid = np.isfinite(depth_m)
    valid &= depth_m >= min_depth_m
    valid &= depth_m <= max_depth_m

    z = depth_m[valid]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    xyz = np.stack((x, y, z), axis=-1).astype(np.float32)

    if not include_rgb:
        return xyz

    rgb = image_to_hwc(sample[ORBBEC_RGB_KEY])[::pixel_stride, ::pixel_stride, :3]
    rgb = rgb[valid].astype(np.float32)
    if rgb.size and rgb.max() <= 1.0 + 1e-6:
        rgb *= 255.0
    rgb = np.clip(rgb, 0.0, 255.0)
    return np.concatenate((xyz, rgb), axis=-1)


def load_gravity_direction(path: Path) -> np.ndarray:
    """Read the simple unit_vector entry without requiring PyYAML."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"unit_vector:\s*\[([^\]]+)\]", text)
    if match is None:
        raise ValueError(f"No gravity_direction.unit_vector found in {path}")
    gravity = np.fromstring(match.group(1), sep=",", dtype=np.float64)
    if gravity.shape != (3,):
        raise ValueError(f"Expected three gravity values in {path}, got {gravity}")
    return gravity / np.linalg.norm(gravity)


def align_with_gravity(point_cloud: np.ndarray, gravity_down: np.ndarray) -> np.ndarray:
    """Rotate camera points to a frame whose positive Z axis points upward."""
    z_up = -gravity_down
    camera_x = np.array([1.0, 0.0, 0.0])
    x_axis = camera_x - np.dot(camera_x, z_up) * z_up
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_up, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    camera_to_gravity = np.stack((x_axis, y_axis, z_up), axis=0)

    result = point_cloud.copy()
    result[:, :3] = result[:, :3] @ camera_to_gravity.T
    return result


def crop_point_cloud(
    point_cloud: np.ndarray,
    crop_min: np.ndarray,
    crop_max: np.ndarray,
) -> np.ndarray:
    if np.any(crop_min >= crop_max):
        raise ValueError(f"Each crop-min value must be smaller than crop-max: {crop_min}, {crop_max}")
    xyz = point_cloud[:, :3]
    mask = np.all(xyz >= crop_min, axis=1)
    mask &= np.all(xyz <= crop_max, axis=1)
    return point_cloud[mask]


def preview_subset(point_cloud: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(point_cloud) <= max_points:
        return point_cloud
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(point_cloud), max_points, replace=False)
    return point_cloud[indices]


def describe(name: str, point_cloud: np.ndarray) -> None:
    print(f"{name}: {len(point_cloud):,} points, shape={point_cloud.shape}")
    if len(point_cloud):
        xyz = point_cloud[:, :3]
        print(f"  xyz min: {xyz.min(axis=0)}")
        print(f"  xyz max: {xyz.max(axis=0)}")


def save_crop_selector_html(
    point_cloud: np.ndarray,
    file_path: Path,
    initial_min: np.ndarray | None = None,
    initial_max: np.ndarray | None = None,
) -> None:
    """Save an offline HTML tool for tuning and exporting axis-aligned crop bounds."""
    from plotly.offline import get_plotlyjs

    xyz = point_cloud[:, :3]
    data_min = xyz.min(axis=0)
    data_max = xyz.max(axis=0)
    if initial_min is None:
        initial_min = data_min
    if initial_max is None:
        initial_max = data_max

    if point_cloud.shape[1] >= 6:
        rgb = np.clip(point_cloud[:, 3:6], 0, 255).astype(np.uint8)
    else:
        span = np.maximum(data_max - data_min, 1e-8)
        rgb = np.clip((xyz - data_min) / span * 255, 0, 255).astype(np.uint8)
    colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]

    payload = json.dumps(
        {
            "x": xyz[:, 0].tolist(),
            "y": xyz[:, 1].tolist(),
            "z": xyz[:, 2].tolist(),
            "colors": colors,
            "dataMin": data_min.tolist(),
            "dataMax": data_max.tolist(),
            "initialMin": np.asarray(initial_min).tolist(),
            "initialMax": np.asarray(initial_max).tolist(),
        },
        separators=(",", ":"),
    )

    controls = []
    axis_names = ("X", "Y", "Z")
    for axis_index, axis_name in enumerate(axis_names):
        axis_min = float(data_min[axis_index])
        axis_max = float(data_max[axis_index])
        step = max((axis_max - axis_min) / 1000.0, 0.0001)
        controls.append(
            f"""
            <div class="axis-control">
              <div class="axis-title">{axis_name}</div>
              <label>min
                <input id="{axis_name.lower()}minNumber" type="number" step="{step:.7f}" value="{initial_min[axis_index]:.7f}">
              </label>
              <input id="{axis_name.lower()}minRange" type="range" min="{axis_min:.7f}" max="{axis_max:.7f}"
                     step="{step:.7f}" value="{initial_min[axis_index]:.7f}">
              <label>max
                <input id="{axis_name.lower()}maxNumber" type="number" step="{step:.7f}" value="{initial_max[axis_index]:.7f}">
              </label>
              <input id="{axis_name.lower()}maxRange" type="range" min="{axis_min:.7f}" max="{axis_max:.7f}"
                     step="{step:.7f}" value="{initial_max[axis_index]:.7f}">
            </div>
            """
        )

    html_template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>LeRobot 点云裁剪区域选择器</title>
  <style>
    body { margin: 0; font-family: sans-serif; color: #222; background: #f5f5f5; }
    #toolbar { padding: 12px 16px; background: white; box-shadow: 0 1px 4px #bbb; }
    #controls { display: grid; grid-template-columns: repeat(3, minmax(260px, 1fr)); gap: 16px; }
    .axis-control { padding: 10px; border: 1px solid #ddd; border-radius: 6px; }
    .axis-title { font-weight: 700; margin-bottom: 5px; }
    label { display: inline-flex; width: 49%; align-items: center; gap: 5px; }
    input[type=number] { width: 120px; }
    input[type=range] { width: 100%; }
    #status { margin-top: 10px; font-weight: 600; }
    #command { width: calc(100% - 24px); margin-top: 8px; padding: 8px; font-family: monospace; }
    button { margin: 8px 8px 0 0; padding: 7px 12px; cursor: pointer; }
    #cropPlot { width: 100%; height: calc(100vh - 285px); min-height: 520px; }
    .hint { color: #555; font-size: 13px; }
    @media (max-width: 900px) { #controls { grid-template-columns: 1fr; } #cropPlot { height: 650px; } }
  </style>
  <script>__PLOTLY_JS__</script>
</head>
<body>
  <div id="toolbar">
    <div class="hint">拖动六个滑块调整重力对齐坐标系中的 AABB。框内保留原色，框外显示灰色背景。</div>
    <div id="controls">__CONTROLS__</div>
    <div id="status"></div>
    <input id="command" readonly>
    <button id="copyButton">复制裁剪参数</button>
    <button id="downloadButton">下载 crop_bounds.json</button>
    <span id="copyStatus"></span>
  </div>
  <div id="cropPlot"></div>
  <script>
    const cloud = __PAYLOAD__;
    const axes = ['x', 'y', 'z'];

    function value(id) { return Number(document.getElementById(id).value); }
    function bounds() {
      return {
        min: axes.map(a => value(a + 'minNumber')),
        max: axes.map(a => value(a + 'maxNumber'))
      };
    }
    function boxCoordinates(lo, hi) {
      const c = [
        [lo[0],lo[1],lo[2]], [hi[0],lo[1],lo[2]], [hi[0],hi[1],lo[2]], [lo[0],hi[1],lo[2]],
        [lo[0],lo[1],hi[2]], [hi[0],lo[1],hi[2]], [hi[0],hi[1],hi[2]], [lo[0],hi[1],hi[2]]
      ];
      const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
      const out = {x: [], y: [], z: []};
      for (const [a,b] of edges) {
        out.x.push(c[a][0], c[b][0], null);
        out.y.push(c[a][1], c[b][1], null);
        out.z.push(c[a][2], c[b][2], null);
      }
      return out;
    }
    function updateCrop() {
      const b = bounds();
      const inside = {x: [], y: [], z: [], colors: []};
      const outside = {x: [], y: [], z: []};
      for (let i = 0; i < cloud.x.length; i++) {
        const keep = cloud.x[i] >= b.min[0] && cloud.x[i] <= b.max[0] &&
                     cloud.y[i] >= b.min[1] && cloud.y[i] <= b.max[1] &&
                     cloud.z[i] >= b.min[2] && cloud.z[i] <= b.max[2];
        const target = keep ? inside : outside;
        target.x.push(cloud.x[i]); target.y.push(cloud.y[i]); target.z.push(cloud.z[i]);
        if (keep) inside.colors.push(cloud.colors[i]);
      }
      const box = boxCoordinates(b.min, b.max);
      Plotly.restyle('cropPlot', {x:[outside.x], y:[outside.y], z:[outside.z]}, [0]);
      Plotly.restyle('cropPlot', {x:[inside.x], y:[inside.y], z:[inside.z], 'marker.color':[inside.colors]}, [1]);
      Plotly.restyle('cropPlot', {x:[box.x], y:[box.y], z:[box.z]}, [2]);
      const validBounds = b.min.every((v, i) => v < b.max[i]);
      document.getElementById('status').textContent = validBounds
        ? `保留 ${inside.x.length.toLocaleString()} / ${cloud.x.length.toLocaleString()} 个预览点`
        : '边界无效：每个 min 必须小于 max';
      const fmt = values => values.map(v => v.toFixed(6)).join(' ');
      document.getElementById('command').value = `--crop-min ${fmt(b.min)} --crop-max ${fmt(b.max)}`;
    }
    function bindPair(axis, side) {
      const range = document.getElementById(axis + side + 'Range');
      const number = document.getElementById(axis + side + 'Number');
      range.addEventListener('input', () => { number.value = range.value; updateCrop(); });
      number.addEventListener('input', () => { range.value = number.value; updateCrop(); });
    }
    for (const axis of axes) { bindPair(axis, 'min'); bindPair(axis, 'max'); }

    const initialBox = boxCoordinates(cloud.initialMin, cloud.initialMax);
    const traces = [
      {type:'scatter3d', mode:'markers', name:'框外背景', x:[], y:[], z:[],
       marker:{size:2, color:'#aaaaaa', opacity:0.10}, hoverinfo:'skip'},
      {type:'scatter3d', mode:'markers', name:'保留区域', x:cloud.x, y:cloud.y, z:cloud.z,
       marker:{size:3, color:cloud.colors, opacity:0.9},
       hovertemplate:'x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>'},
      {type:'scatter3d', mode:'lines', name:'裁剪框', x:initialBox.x, y:initialBox.y, z:initialBox.z,
       line:{color:'#ff3030', width:7}, hoverinfo:'skip'}
    ];
    Plotly.newPlot('cropPlot', traces, {
      margin:{l:0,r:0,b:0,t:0},
      scene:{aspectmode:'data', xaxis:{title:'X (m)'}, yaxis:{title:'Y (m)'}, zaxis:{title:'Z (m)'}},
      legend:{x:0.01,y:0.99}
    }, {responsive:true}).then(updateCrop);

    document.getElementById('copyButton').addEventListener('click', async () => {
      const text = document.getElementById('command').value;
      await navigator.clipboard.writeText(text);
      document.getElementById('copyStatus').textContent = '已复制，把这行参数发给 Codex 即可。';
    });
    document.getElementById('downloadButton').addEventListener('click', () => {
      const b = bounds();
      const blob = new Blob([JSON.stringify({crop_min:b.min, crop_max:b.max}, null, 2)], {type:'application/json'});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob); link.download = 'crop_bounds.json'; link.click();
      URL.revokeObjectURL(link.href);
    });
  </script>
</body>
</html>
"""
    html = (
        html_template.replace("__PLOTLY_JS__", get_plotlyjs())
        .replace("__CONTROLS__", "".join(controls))
        .replace("__PAYLOAD__", payload)
    )
    file_path.write_text(html, encoding="utf-8")


def get_visualizer_class():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_project = repo_root / "visualizer"
    sys.path.insert(0, str(visualizer_project))
    try:
        from visualizer import Visualizer
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import DP3 visualizer. Install its dependencies and package with:\n"
            "  pip install flask plotly matplotlib termcolor\n"
            "  pip install -e visualizer"
        ) from exc
    return Visualizer


def make_clean_dataset_view(dataset_root: Path):
    """Hide macOS AppleDouble files without modifying the source dataset."""
    ignored_names = {".DS_Store"}
    has_sidecars = any(
        path.name.startswith("._") or path.name in ignored_names
        for path in dataset_root.rglob("*")
    )
    if not has_sidecars:
        return dataset_root, None

    temporary_dir = tempfile.TemporaryDirectory(prefix="lerobot-clean-")
    clean_root = Path(temporary_dir.name)
    for source in dataset_root.rglob("*"):
        if source.name.startswith("._") or source.name in ignored_names:
            continue
        relative = source.relative_to(dataset_root)
        destination = clean_root / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source.resolve())

    print(f"Using a temporary clean dataset view that ignores macOS sidecar files: {clean_root}")
    return clean_root, temporary_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/hinyeun_glue_0714_rgbd")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=1.8)
    parser.add_argument("--xyz-only", action="store_true", help="Do not attach registered RGB colors")
    parser.add_argument(
        "--gravity-align",
        action="store_true",
        help="Rotate points so positive Z points upward using meta/orbbec_gravity.yaml",
    )
    parser.add_argument("--crop-min", type=float, nargs=3, metavar=("XMIN", "YMIN", "ZMIN"))
    parser.add_argument("--crop-max", type=float, nargs=3, metavar=("XMAX", "YMAX", "ZMAX"))
    parser.add_argument("--preview-points", type=int, default=30_000)
    parser.add_argument(
        "--crop-selector",
        action="store_true",
        help="Save an interactive HTML crop-bound selector with live point counts",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("pointcloud_preview"))
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Also serve the selected cloud at http://127.0.0.1:5000 (blocks until stopped)",
    )
    args = parser.parse_args()

    if args.pixel_stride < 1:
        parser.error("--pixel-stride must be at least 1")
    if args.frame_index < 0:
        parser.error("--frame-index must be non-negative")
    if (args.crop_min is None) != (args.crop_max is None):
        parser.error("--crop-min and --crop-max must be provided together")
    return args


def main() -> None:
    args = parse_args()
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is required to decode this dataset. Install the version documented by the dataset:\n"
            "  pip install 'lerobot[dataset]==0.6.0' 'av==15.1.0'"
        ) from exc

    dataset_root = args.dataset_root.expanduser().resolve()
    loader_root, clean_view = make_clean_dataset_view(dataset_root)
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=loader_root,
        video_backend="pyav",
    )
    if args.frame_index >= len(dataset):
        raise IndexError(f"frame-index {args.frame_index} is outside dataset length {len(dataset)}")

    sample = dataset[args.frame_index]
    point_cloud = reconstruct_point_cloud(
        sample=sample,
        pixel_stride=args.pixel_stride,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        include_rgb=not args.xyz_only,
    )

    if args.gravity_align:
        gravity_path = dataset_root / "meta" / "orbbec_gravity.yaml"
        point_cloud = align_with_gravity(point_cloud, load_gravity_direction(gravity_path))

    describe("reconstructed", point_cloud)
    selected = point_cloud
    if args.crop_min is not None:
        selected = crop_point_cloud(
            point_cloud,
            np.asarray(args.crop_min, dtype=np.float32),
            np.asarray(args.crop_max, dtype=np.float32),
        )
        describe("cropped", selected)
        if not len(selected):
            raise ValueError("The crop is empty; widen or correct the crop bounds")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"frame_{args.frame_index:06d}"
    Visualizer = get_visualizer_class()

    raw_preview = preview_subset(point_cloud, args.preview_points, args.seed)
    raw_html = args.output_dir / f"{prefix}_reconstructed.html"
    Visualizer().save_visualization_to_file(raw_preview, str(raw_html))
    print(f"Saved reconstructed preview: {raw_html.resolve()}")

    if args.crop_selector:
        selector_html = args.output_dir / f"{prefix}_crop_selector.html"
        selector_min = None if args.crop_min is None else np.asarray(args.crop_min, dtype=np.float32)
        selector_max = None if args.crop_max is None else np.asarray(args.crop_max, dtype=np.float32)
        save_crop_selector_html(raw_preview, selector_html, selector_min, selector_max)
        print(f"Saved interactive crop selector: {selector_html.resolve()}")

    if args.crop_min is not None:
        cropped_preview = preview_subset(selected, args.preview_points, args.seed)
        cropped_html = args.output_dir / f"{prefix}_cropped.html"
        Visualizer().save_visualization_to_file(cropped_preview, str(cropped_html))
        print(f"Saved cropped preview: {cropped_html.resolve()}")

    npy_path = args.output_dir / f"{prefix}_selected.npy"
    np.save(npy_path, selected)
    print(f"Saved selected point cloud: {npy_path.resolve()}")

    if args.serve:
        served = preview_subset(selected, args.preview_points, args.seed)
        print("Serving point cloud at http://127.0.0.1:5000 (Ctrl+C to stop)")
        Visualizer().visualize_pointcloud(served)

    # Keep the temporary clean dataset view alive until all decoding is done.
    del clean_view


if __name__ == "__main__":
    main()
