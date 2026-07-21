"""DP3 dataset adapter for the right-arm HINYEUN glue demonstrations.

The converted Zarr stores arrays under ``data/`` using the names
``point_cloud``, ``state`` and ``action``.  DP3 expects observations named
``point_cloud`` and ``agent_pos``, so this class performs that small naming
adaptation and samples fixed-length sequences without crossing episode
boundaries.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer


class HinyeunGlueDataset(BaseDataset):
    """Load cropped XYZ point clouds and right-arm state/action trajectories.

    Expected per-frame shapes in the converted Zarr are:

    - ``point_cloud``: ``(1024, 3)`` XYZ in the gravity-aligned crop frame
    - ``state``: ``(8,)`` right arm's seven joints plus gripper width
    - ``action``: ``(9,)`` right arm, gripper command, and dispenser command
    """

    expected_point_cloud_shape = (1024, 3)
    expected_state_shape = (8,)
    expected_action_shape = (9,)

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.1,
        max_train_episodes: int | None = None,
        task_name: str | None = None,
    ):
        super().__init__()

        self.task_name = task_name
        zarr_path = self._resolve_zarr_path(zarr_path)
        # RGB images are intentionally absent: DP3 consumes XYZ + robot state.
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=["state", "action", "point_cloud"],
        )
        self._validate_replay_buffer()

        # Split whole episodes, not individual frames, to prevent train/val
        # leakage between neighboring samples from the same demonstration.
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(
            mask=~val_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    @staticmethod
    def _resolve_zarr_path(zarr_path: str) -> str:
        """Resolve task-relative paths even when ``train.py`` changes cwd."""
        path = Path(zarr_path).expanduser()
        if path.is_absolute() or path.exists():
            return str(path)

        # ``hinyeun_glue_dataset.py`` lives at
        # <project>/diffusion_policy_3d/dataset/, while task YAML paths are
        # relative to <project>.  The upstream train entry point changes the
        # working directory before Hydra starts, so cwd alone is unreliable.
        project_relative_path = Path(__file__).resolve().parents[2] / path
        if project_relative_path.exists():
            return str(project_relative_path)

        raise FileNotFoundError(
            f"HINYEUN Zarr dataset not found at {path!s} or "
            f"{project_relative_path!s}. Run the conversion script first or "
            "set task.dataset.zarr_path explicitly."
        )

    def _validate_replay_buffer(self) -> None:
        expected_shapes = {
            "point_cloud": self.expected_point_cloud_shape,
            "state": self.expected_state_shape,
            "action": self.expected_action_shape,
        }
        for key, expected_shape in expected_shapes.items():
            actual_shape = tuple(self.replay_buffer[key].shape[1:])
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Unexpected {key} shape {actual_shape}; expected {expected_shape}. "
                    "Check that the right-arm DP3 conversion output was selected."
                )

    def get_validation_dataset(self) -> "HinyeunGlueDataset":
        val_dataset = copy.copy(self)
        val_dataset.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_dataset.train_mask = ~self.train_mask
        return val_dataset

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        # Each last dimension is normalized independently: XYZ, eight state
        # channels, and nine action channels therefore keep separate scales.
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        actions = np.asarray(self.replay_buffer["action"][:], dtype=np.float32)
        return torch.from_numpy(actions)

    def __len__(self) -> int:
        return len(self.sampler)

    @staticmethod
    def _sample_to_data(sample: dict[str, np.ndarray]) -> dict:
        return {
            "obs": {
                "point_cloud": sample["point_cloud"].astype(np.float32, copy=False),
                # DP3 calls the proprioceptive state "agent_pos".
                "agent_pos": sample["state"].astype(np.float32, copy=False),
            },
            "action": sample["action"].astype(np.float32, copy=False),
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(index)
        return dict_apply(self._sample_to_data(sample), torch.from_numpy)
