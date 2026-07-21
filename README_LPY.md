# README_LPY — HINYEUN Glue 右臂 DP3

本文档记录本仓库中与 HINYEUN 涂胶数据相关的**点云裁剪 / 转换**与**离线 DP3 训练**流程。

---

## 1. 相关文件

| 项目 | 路径 |
|------|------|
| 点云预览 / 裁剪调参 | `scripts/visualize_lerobot_pointcloud.py` |
| LeRobot → DP3 Zarr 转换 | `scripts/convert_hinyeun_lerobot_to_dp3.py` |
| 训练用 Zarr | `3D-Diffusion-Policy/data/hinyeun_glue_right_dp3.zarr` |
| Task 配置 | `3D-Diffusion-Policy/diffusion_policy_3d/config/task/hinyeun_glue.yaml` |
| Dataset 类 | `3D-Diffusion-Policy/diffusion_policy_3d/dataset/hinyeun_glue_dataset.py` |
| 主配置 | `3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml` |

### 数据维度（须与 Zarr 一致）

- `point_cloud`: `[1024, 3]`（重力对齐 XYZ，无 RGB）
- `agent_pos` / `state`: `[8]`（右臂 7 关节 + 夹爪宽度）
- `action`: `[9]`（7 关节 + 夹爪指令 + 点胶指令）
- 推荐时序：`horizon=16`, `n_obs_steps=2`, `n_action_steps=4`（约 10 Hz）
- `env_runner: null`（目前仅离线训练）

当前完整 Zarr 元信息摘要（见 `.zattrs`）：约 20272 帧、源数据 30 Hz → 10 Hz、`sampling=fps`、`voxel_size=0.005`。

---

## 2. 点云裁剪在哪里

裁剪相关逻辑**不在训练 Dataset 里**（训练时直接读已裁好的 Zarr），而是在转换链路里：

1. **调参 / 可视化**：`scripts/visualize_lerobot_pointcloud.py`
   - `reconstruct_point_cloud`：Orbbec 深度反投影
   - `load_gravity_direction` / `align_with_gravity`：用 `meta/orbbec_gravity.yaml` 把相机系转到 Z 朝上
   - `crop_point_cloud`：轴对齐 AABB 裁剪
   - `save_crop_selector_html`：交互式 HTML，拖滑块调 `crop_min` / `crop_max`，可复制参数或下载 `crop_bounds.json`
2. **批量写入 Zarr**：`scripts/convert_hinyeun_lerobot_to_dp3.py`
   - 复用上面的对齐与裁剪函数
   - 默认裁剪框：
     - `crop_min = (-0.222791, 0.238951, -0.282919)`
     - `crop_max = (0.469885, 0.774905, 0.075385)`
   - 之后再做体素下采样 + FPS/uniform，固定到 1024 点

处理顺序（转换时）：

```text
深度反投影 → 重力对齐 → AABB 裁剪 → voxel downsample → FPS/uniform → (T, 1024, 3)
```

### 2.1 用交互工具调裁剪框

```bash
# 在仓库根目录；需已安装 lerobot / plotly 等依赖
python scripts/visualize_lerobot_pointcloud.py \
  --dataset-root /home/descfly/lpy/hinyeun_glue_0714_lerobot_rgbd \
  --repo-id local/hinyeun_glue_0714_rgbd \
  --frame-index 0 \
  --gravity-align \
  --crop-selector \
  --output-dir pointcloud_preview
```

打开生成的 `pointcloud_preview/frame_XXXXXX_crop_selector.html`，调好后复制 `--crop-min ... --crop-max ...`，再喂给转换脚本。

也可直接带已知框预览裁剪结果：

```bash
python scripts/visualize_lerobot_pointcloud.py \
  --dataset-root /home/descfly/lpy/hinyeun_glue_0714_lerobot_rgbd \
  --gravity-align \
  --crop-min -0.222791 0.238951 -0.282919 \
  --crop-max 0.469885 0.774905 0.075385 \
  --output-dir pointcloud_preview
```

### 2.2 转换完整数据集

```bash
python scripts/convert_hinyeun_lerobot_to_dp3.py \
  --dataset-root /home/descfly/lpy/hinyeun_glue_0714_lerobot_rgbd \
  --output /home/descfly/lpy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/hinyeun_glue_right_dp3.zarr \
  --repo-id local/hinyeun_glue_0714_rgbd \
  --target-fps 10 \
  --num-points 1024 \
  --crop-min -0.222791 0.238951 -0.282919 \
  --crop-max 0.469885 0.774905 0.075385 \
  --voxel-size 0.005 \
  --sampling fps \
  --overwrite
```

冒烟可用 `--max-episodes N`（已有 `hinyeun_glue_right_dp3_smoke.zarr`）。

常用转换参数：

| 参数 | 默认 | 含义 |
|------|------|------|
| `--target-fps` | 10 | 从源 fps 均匀抽帧（须整除） |
| `--num-points` | 1024 | 每帧固定点数 |
| `--pixel-stride` | 4 | 深度反投影像素步长 |
| `--min-depth-m` / `--max-depth-m` | 0.15 / 1.8 | 深度有效范围 |
| `--crop-min` / `--crop-max` | 见上 | 重力对齐系下 AABB |
| `--voxel-size` | 0.005 | FPS 前体素下采样 |
| `--sampling` | `fps` | `fps` 或 `uniform` |

---

## 3. 训练环境

```bash
conda activate dp3
cd /home/descfly/lpy/3D-Diffusion-Policy/3D-Diffusion-Policy
```

### 3.1 移植 `dp3` conda 环境到其他电脑

官方 `INSTALL.md` 面向仿真全栈（Python 3.8 + MuJoCo 等）。本机实际用于 **离线训练 HINYEUN zarr** 的 `dp3` 更精简：

| 项 | 本机快照 |
|----|----------|
| Python | 3.10 |
| torch | `2.7.1+cu128` |
| 本地 editable | `3D-Diffusion-Policy/`、`third_party/pytorch3d_simplified/` |
| GPU 参考 | RTX 5090，driver ~580 |

**不要**直接用 `conda env export` / `conda list --explicit` 整包搬迁：路径、CUDA wheel、editable 包都会坏。应写两类文件 + 一条安装脚本：

| 文件 | 作用 |
|------|------|
| `environment_dp3.yml` | 只建空 conda 环境（Python 3.10 + pip） |
| `requirements_dp3.txt` | 训练用的 pip 依赖（**不含** torch） |
| `scripts/setup_dp3_env.sh` | 按 CUDA 通道装 torch，再装依赖与本地 editable |

在新机器上：

```bash
# 1) 拷贝/克隆本仓库（含 zarr 数据若需要）
git clone <your-fork-or-path> 3D-Diffusion-Policy
cd 3D-Diffusion-Policy

# 2) 创建 conda 环境
conda env create -f environment_dp3.yml
conda activate dp3

# 3) 安装 torch + 依赖 + 本地包
bash scripts/setup_dp3_env.sh
```

若目标机器驱动较旧、不支持 CUDA 12.8 wheel，改 CUDA 通道，例如：

```bash
TORCH_CUDA=cu121 bash scripts/setup_dp3_env.sh
```

到 [pytorch.org](https://pytorch.org/get-started/locally/) 选与本机驱动匹配的 `cu118` / `cu121` / `cu124` / `cu128`。

**说明：**

- 本流程只覆盖 **DP3 离线训练**。点云转换 / LeRobot 解码在本机是另一个环境 `lerobot`，需要另装（`lerobot[dataset]`、`av` 等），不要塞进 `dp3`。
- 若还要跑官方仿真（Adroit / MetaWorld），仍按仓库根目录 `INSTALL.md`，与上面这套不是同一条路径。
- 换机器后请再跑一次冒烟训练（见下文），确认 `torch.cuda.is_available()` 为 `True`。

### 默认训练超参（`dp3.yaml`）

- `num_epochs: 3000`
- `checkpoint_every: 200`（且需 `checkpoint.save_ckpt=true` 才会周期性存 ckpt）
- `val_every: 1`
- 默认 `batch_size=128`；本任务实测常用覆盖为 `32`

总耗时 ≈ `3000 × (单 epoch 耗时)`，以终端进度条为准。

---

## 4. 启动训练

**必须**指定 `--config-name=dp3`，否则 Hydra 会报：

`Could not override 'task'. No match in the defaults list.`

`train.py` 会把工作目录切到仓库更上层，相对路径 `data/...` 容易找不到，**建议用绝对路径**指定 zarr。

```bash
conda activate dp3
cd /home/descfly/lpy/3D-Diffusion-Policy/3D-Diffusion-Policy

python train.py --config-name=dp3 \
  task=hinyeun_glue \
  horizon=16 \
  n_obs_steps=2 \
  n_action_steps=4 \
  task.dataset.zarr_path=/home/descfly/lpy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/hinyeun_glue_right_dp3.zarr \
  dataloader.batch_size=32 \
  dataloader.num_workers=4 \
  val_dataloader.batch_size=32 \
  val_dataloader.num_workers=4 \
  training.device=cuda:0 \
  training.resume=false \
  logging.mode=offline \
  checkpoint.save_ckpt=true \
  exp_name=hinyeun-glue-dp3
```

### 粘贴注意

多行命令时，每行末尾只能是 `\`，后面不能跟空格或别的字符。从聊天复制时容易把下一行开头粘进上一行（例如 `\2`、`\4`），导致参数错乱。

### 从 checkpoint 恢复

若 `checkpoints/latest.ckpt` 已存在，将 `training.resume=false` 改为 `true`，并尽量固定同一输出目录：

```text
hydra.run.dir=/home/descfly/lpy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/hinyeun-glue-dp3
```

### 输出位置

- 日志 / Hydra / wandb：`3D-Diffusion-Policy/data/outputs/<exp_name>/`
- Checkpoint：同目录 `checkpoints/`
- 历史参考：`hinyeun-glue-dp3/`、`hinyeun-glue-dp3-smoke/`

### 冒烟测试

```bash
python train.py --config-name=dp3 \
  task=hinyeun_glue \
  horizon=16 \
  n_obs_steps=2 \
  n_action_steps=4 \
  task.dataset.zarr_path=/home/descfly/lpy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/hinyeun_glue_right_dp3_smoke.zarr \
  dataloader.batch_size=8 \
  dataloader.num_workers=0 \
  val_dataloader.batch_size=8 \
  val_dataloader.num_workers=0 \
  training.device=cuda:0 \
  training.num_epochs=1 \
  training.max_train_steps=10 \
  training.resume=false \
  logging.mode=offline \
  checkpoint.save_ckpt=false \
  exp_name=hinyeun-glue-dp3-smoke
```

---

## 5. 常见问题

1. **`Could not override 'task'`** — 缺少 `--config-name=dp3`。
2. **找不到 zarr** — 用 `task.dataset.zarr_path=` 绝对路径。
3. **命令粘贴损坏** — 检查 `\` 后无多余字符；不要粘进 shell 提示符。
4. **完整 Hydra 堆栈** — `export HYDRA_FULL_ERROR=1`。
5. **裁剪框为空 / 点太少** — 先用 `--crop-selector` 在重力对齐坐标系下调 AABB，再重跑转换。
