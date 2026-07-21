# <a href="https://3d-diffusion-policy.github.io">3D Diffusion Policy</a>

<a href="https://3d-diffusion-policy.github.io"><strong>Project Page</strong></a>
  |
  <a href="https://arxiv.org/abs/2403.03954"><strong>arXiv</strong></a>
  |
  <a href="https://x.com/ZeYanjie/status/1765414787775963232?s=20"><strong>Twitter</strong></a> | <a href="https://1drv.ms/u/s!Ag5QsBIFtRnTlFWqYWtS2wMMPKNX?e=dw8hsS"><strong>Data</strong></a>

  <a href="https://yanjieze.com/">Yanjie Ze*</a>,
  <a href="https://www.gu-zhang.com/">Gu Zhang*</a>,
  <a href="https://zkangning.github.io">Kangning Zhang</a>,
  <a href="https://github.com/pummmmpkin">Chenyuan Hu</a>,
  <a href="https://wang-muhan.github.io/">Muhan Wang</a>,
  <a href="http://hxu.rocks/">Huazhe Xu</a>


**Robotics: Science and Systems (RSS) 2024**

<div align="center">
  <img src="DP3.png" alt="dp3" width="100%">
</div>

**3D Diffusion Policy (DP3)** 是一种通用的视觉模仿学习算法，将 3D 视觉表征与 Diffusion Policy 结合，在仿真与真实机器人任务中都表现突出，同时兼顾高维 / 低维控制，并具备实用的推理速度。

**使用小提示：**
- 尝试更长的 [prediction horizon](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/b147695af3ecb90101745fe9778ade2f86f23a46/3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml#L10) 与 [action horizon](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/b147695af3ecb90101745fe9778ade2f86f23a46/3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml#L12)（如 8/16/32），通常效果更好。
- 动作空间优先使用**全局位置**，而不是相对位置。

**本仓库额外内容（HINYEUN 涂胶）：** 已加入 LeRobot → DP3 Zarr 转换、点云裁剪与离线训练流程，详见下方「HINYEUN Glue 右臂 DP3」章节，完整笔记也可直接看 [README_LPY.md](README_LPY.md)。

**社区应用与扩展（节选）：**
- [arXiv 2025.10](https://lei-kun.github.io/RL-100/) *RL-100*：离线 + 在线 RL 微调后，DP3 操作成功率可达 100%。
- [arXiv 2025.09](https://arxiv.org/abs/2509.01819) *ManiFlow*：用新 backbone 与 flow matching 改进 DP3。
- [arXiv 2025.03](https://arxiv.org/abs/2503.07511) *PointVLA*：使用 [iDP3 Encoder](https://github.com/YanjieZe/Improved-3D-Diffusion-Policy) 训练 3D VLA。
- [arXiv 2024.10](https://arxiv.org/abs/2410.10803) *Generalizable Humanoid Manipulation*：改进版 DP3 在人形操作与跨场景泛化上表现突出。
- 更多相关工作见原作者上游仓库 README。

---

# 📊 DP3 Benchmark

**仿真环境：** 本仓库提供 `Adroit`、`DexArt`、`MetaWorld` 的灵巧操作环境与专家策略（共 3+4+50=57 个任务），并已接入深度 / 点云等 3D 模态。

**真机数据：** 官方示例数据见 [此处](https://drive.google.com/file/d/1G5MP6Nzykku9sDDdzy7tlRqMBnKb253O/view?usp=sharing)。

**算法配置：**
- DP3：`dp3.yaml`（论文主方法；A40 上约 10G 显存、约 3 小时）
- Simple DP3：`simple_dp3.yaml`（训练更快约 1–2 小时，推理约 **25 FPS**，性能损失较小，更适合机器人落地实验）

---

# 💻 安装

完整安装见 [INSTALL.md](INSTALL.md)。  
安装常见问题见 [ERROR_CATCH.md](ERROR_CATCH.md)。

若只做 **HINYEUN Zarr 离线训练**（不跑官方仿真全栈），可用本仓库精简环境：

```bash
conda env create -f environment_dp3.yml
conda activate dp3
bash scripts/setup_dp3_env.sh
```

说明见下方 HINYEUN 章节，或 [README_LPY.md](README_LPY.md)。

---

# 📚 数据

可用仓库提供的专家策略自行生成演示数据，生成结果默认在 `$YOUR_REPO_PATH/3D-Diffusion-Policy/data/`。

- 下载 Adroit RL experts：[OneDrive](https://1drv.ms/u/s!Ag5QsBIFtRnTlFWqYWtS2wMMPKNX?e=dw8hsS) 或 [GoogleDrive](https://drive.google.com/file/d/1iNkSrLD_N4NrezLx58L1YoBBqYYg-33u/view?usp=sharing)，解压后将 `ckpts` 放到 `$YOUR_REPO_PATH/third_party/VRL3/`。
- 下载 DexArt assets：[Google Drive](https://drive.google.com/file/d/1DxRfB4087PeM3Aejd6cR-RQVgOKdNrL4/view?usp=sharing)，将 `assets` 放到 `$YOUR_REPO_PATH/third_party/dexart-release/`。

**注意：** 自行生成的演示与论文数值可能略有差异，这在模仿学习中很常见。若遇到质量较差的演示，请重新生成，无需为此单独开 issue。

---

# 🛠️ 使用方法

生成演示、训练、评估脚本都在 `scripts/`。结果默认用 `wandb` 记录，首次使用请先 `wandb login`。

1. **生成演示**（示例：Adroit hammer）
   ```bash
   bash scripts/gen_demonstration_adroit.sh hammer
   ```
   数据会自动保存到 `3D-Diffusion-Policy/data/`。

2. **训练并评估策略**
   ```bash
   bash scripts/train_policy.sh dp3 adroit_hammer 0112 0 0
   ```
   默认会保存 checkpoint（可在脚本中关闭）。

3. **评估已保存策略 / 推理部署**
   ```bash
   bash scripts/eval_policy.sh dp3 adroit_hammer 0112 0 0
   ```
   **说明：** 评估脚本主要用于部署 / 推理；论文式 benchmark 请以训练过程中 wandb 记录为准。

---

# 🤖 真机（官方示例）

**硬件：** Franka + Allegro Hand + **L515** RealSense（不建议 D435，点云质量过低可能导致 DP3 失败）等，详见上游说明。

**每条真机演示（长度 T）字段约定：**
1. `point_cloud`: `(T, Np, 6)`，即 `[x, y, z, r, g, b]`。**强烈建议裁掉桌面 / 背景，只保留有效点云。**
2. `image`: `(T, H, W, 3)`
3. `depth`: `(T, H, W)`
4. `agent_pos`: `(T, Nd)`（官方灵巧手任务 Nd=22）
5. `action`: `(T, Nd)`（机械臂相对末端位姿 + 灵巧手相对关节角）

训练前需按论文做点云裁剪与 FPS 下采样。可参考 [`scripts/convert_real_robot_data.py`](scripts/convert_real_robot_data.py)。  
用官方真机数据训练示例：

```bash
bash scripts/train_policy.sh dp3 realdex_drill 0112 0 0
```

真机部署代码可参考 [iDP3](https://github.com/YanjieZe/Improved-3D-Diffusion-Policy)。

---

# 🔍 点云可视化

```bash
cd visualizer
pip install -e .
```

```python
import visualizer
your_pointcloud = ...  # numpy, shape (N, 3) 或 (N, 6)
visualizer.visualize_pointcloud(your_pointcloud)
```

会在浏览器中打开点云可视化页面，便于无头机器调试。

---

# 🦾 接入你自己的任务

1. 为任务写环境 wrapper（参考 `3D-Diffusion-Policy/diffusion_policy_3d/env/adroit`）。
2. 添加 env runner（参考 `env_runner/`）。
3. 准备专家数据（可参考 `third_party/VRL3/src/gen_demonstration.py`）。
4. 添加 Dataset（参考 `dataset/`）。
5. 在 `config/task` 下添加任务 yaml。
6. 用 `scripts/train_policy.sh` 训练与评估。

---

# 🧪 HINYEUN Glue 右臂 DP3

本节汇总本仓库针对 **HINYEUN 涂胶** 的点云裁剪 / LeRobot→Zarr 转换与 **离线训练** 流程。更完整的笔记见 [README_LPY.md](README_LPY.md)（该文件保持独立，不会被本 README 替代）。

## 相关文件

| 项目 | 路径 |
|------|------|
| 点云预览 / 裁剪调参 | `scripts/visualize_lerobot_pointcloud.py` |
| LeRobot → DP3 Zarr 转换 | `scripts/convert_hinyeun_lerobot_to_dp3.py` |
| 训练用 Zarr | `3D-Diffusion-Policy/data/hinyeun_glue_right_dp3.zarr` |
| Task 配置 | `3D-Diffusion-Policy/diffusion_policy_3d/config/task/hinyeun_glue.yaml` |
| Dataset 类 | `3D-Diffusion-Policy/diffusion_policy_3d/dataset/hinyeun_glue_dataset.py` |
| 主配置 | `3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml` |
| 精简训练环境 | `environment_dp3.yml` / `requirements_dp3.txt` / `scripts/setup_dp3_env.sh` |

### 数据维度（须与 Zarr 一致）

- `point_cloud`: `[1024, 3]`（重力对齐 XYZ，无 RGB）
- `agent_pos` / `state`: `[8]`（右臂 7 关节 + 夹爪宽度）
- `action`: `[9]`（7 关节 + 夹爪指令 + 点胶指令）
- 推荐时序：`horizon=16`, `n_obs_steps=2`, `n_action_steps=4`（约 10 Hz）
- `env_runner: null`（目前仅离线训练）

完整 Zarr 约 20272 帧；源数据 30 Hz → 10 Hz；`sampling=fps`；`voxel_size=0.005`。

## 点云裁剪在哪里

裁剪**不在**训练 Dataset 里（训练直接读已裁好的 Zarr），而在转换链路：

1. **调参 / 可视化**：`scripts/visualize_lerobot_pointcloud.py`
   - Orbbec 深度反投影 → 重力对齐 → AABB 裁剪
   - 交互式 HTML 调 `crop_min` / `crop_max`
2. **批量写 Zarr**：`scripts/convert_hinyeun_lerobot_to_dp3.py`
   - 默认裁剪框：
     - `crop_min = (-0.222791, 0.238951, -0.282919)`
     - `crop_max = (0.469885, 0.774905, 0.075385)`
   - 再做体素下采样 + FPS/uniform，固定到 1024 点

处理顺序：

```text
深度反投影 → 重力对齐 → AABB 裁剪 → voxel downsample → FPS/uniform → (T, 1024, 3)
```

### 交互调裁剪框

```bash
python scripts/visualize_lerobot_pointcloud.py \
  --dataset-root /path/to/hinyeun_glue_0714_lerobot_rgbd \
  --repo-id local/hinyeun_glue_0714_rgbd \
  --frame-index 0 \
  --gravity-align \
  --crop-selector \
  --output-dir pointcloud_preview
```

打开 `pointcloud_preview/frame_XXXXXX_crop_selector.html`，调好后把 `--crop-min` / `--crop-max` 传给转换脚本。

### 转换完整数据集

```bash
python scripts/convert_hinyeun_lerobot_to_dp3.py \
  --dataset-root /path/to/hinyeun_glue_0714_lerobot_rgbd \
  --output /path/to/3D-Diffusion-Policy/data/hinyeun_glue_right_dp3.zarr \
  --repo-id local/hinyeun_glue_0714_rgbd \
  --target-fps 10 \
  --num-points 1024 \
  --crop-min -0.222791 0.238951 -0.282919 \
  --crop-max 0.469885 0.774905 0.075385 \
  --voxel-size 0.005 \
  --sampling fps \
  --overwrite
```

冒烟可用 `--max-episodes N`。常用参数：`--target-fps`、`--num-points`、`--pixel-stride`、`--min-depth-m` / `--max-depth-m`、`--crop-min` / `--crop-max`、`--voxel-size`、`--sampling`。

## 训练环境（离线 HINYEUN）

官方 `INSTALL.md` 面向仿真全栈（Python 3.8 + MuJoCo 等）。本仓库用于离线训练的 `dp3` 环境更精简（参考：Python 3.10、`torch 2.7.1+cu128`）。

```bash
conda env create -f environment_dp3.yml
conda activate dp3
bash scripts/setup_dp3_env.sh
# 若 CUDA 较旧：TORCH_CUDA=cu121 bash scripts/setup_dp3_env.sh
```

- 本流程只覆盖 **DP3 离线训练**；LeRobot 点云转换请用单独环境，不要塞进 `dp3`。
- 若要跑官方仿真（Adroit / MetaWorld），仍按 `INSTALL.md`。

默认超参（`dp3.yaml`）：`num_epochs=3000`，`checkpoint_every=200`（需 `checkpoint.save_ckpt=true`），本任务常用 `batch_size=32`。

## 启动训练

**必须**指定 `--config-name=dp3`，否则 Hydra 会报 `Could not override 'task'`。  
`train.py` 会切换工作目录，**建议用绝对路径**指定 zarr：

```bash
conda activate dp3
cd /path/to/3D-Diffusion-Policy/3D-Diffusion-Policy

python train.py --config-name=dp3 \
  task=hinyeun_glue \
  horizon=16 \
  n_obs_steps=2 \
  n_action_steps=4 \
  task.dataset.zarr_path=/path/to/hinyeun_glue_right_dp3.zarr \
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

多行命令时，每行末尾 `\` 后不能有空格。恢复训练时设 `training.resume=true`，并尽量固定 `hydra.run.dir=...`。

输出默认在 `3D-Diffusion-Policy/data/outputs/<exp_name>/`（含 `checkpoints/`）。

### 冒烟测试

```bash
python train.py --config-name=dp3 \
  task=hinyeun_glue \
  horizon=16 \
  n_obs_steps=2 \
  n_action_steps=4 \
  task.dataset.zarr_path=/path/to/hinyeun_glue_right_dp3_smoke.zarr \
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

### 常见问题

1. `Could not override 'task'` → 缺少 `--config-name=dp3`
2. 找不到 zarr → 用 `task.dataset.zarr_path=` 绝对路径
3. 命令粘贴损坏 → 检查 `\` 后无多余字符
4. 需要完整 Hydra 堆栈 → `export HYDRA_FULL_ERROR=1`
5. 裁剪后点太少 → 先用 `--crop-selector` 在重力对齐坐标系下调 AABB

---

# 🏷️ License

本仓库基于 MIT 协议发布，详见 [LICENSE](LICENSE)。

# 😺 Acknowledgement

代码主要基于：[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)、[DexMV](https://github.com/yzqin/dexmv-sim)、[DexArt](https://github.com/Kami-code/dexart-release)、[VRL3](https://github.com/microsoft/VRL3)、[DAPG](https://github.com/aravindr93/hand_dapg)、[DexDeform](https://github.com/sizhe-li/DexDeform)、[RL3D](https://github.com/YanjieZe/rl3d)、[GNFactor](https://github.com/YanjieZe/GNFactor)、[H-InDex](https://github.com/YanjieZe/H-InDex)、[MetaWorld](https://github.com/Farama-Foundation/Metaworld)、[BEE](https://jity16.github.io/BEE/)、[Bi-DexHands](https://github.com/PKU-MARL/DexterousHands)、[HORA](https://github.com/HaozhiQi/hora) 等开源项目。感谢原作者与社区贡献。

原项目问题可联系 [Yanjie Ze](https://yanjieze.com)。

# 📝 Citation

如果本工作对你有帮助，请引用：

```
@inproceedings{Ze2024DP3,
	title={3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations},
	author={Yanjie Ze and Gu Zhang and Kangning Zhang and Chenyuan Hu and Muhan Wang and Huazhe Xu},
	booktitle={Proceedings of Robotics: Science and Systems (RSS)},
	year={2024}
}
```
