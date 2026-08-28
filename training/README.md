# training/ —— BC 策略训练与评估工具

封装 Isaac Lab 官方 robomimic 训练/评估脚本，固化默认参数 + 自动日志归档，实现一键训练/评估/诊断。

## 快速开始

```bash
cd /root/gpufree-data/isaac_sim_learning/synthetic-manipulation-motion-generation

# 训练（默认训满 2000 epoch，后台跑）
nohup python training/train.py > /dev/null 2>&1 &
tail -f training/logs/train_*.log

# 评估（自动找最新 checkpoint，10 次 rollout 输出成功率）
python training/evaluate.py

# 诊断（文字可视化机械臂一次 rollout 的行为）
python training/diagnose.py --headless

# 分析训练日志（画 Loss/梯度/耗时等指标报告图）
python training/analyze.py

# 录像（成功/失败各录一条 MP4；--enable_cameras 必须加，否则相机不渲染）
python training/record.py --headless --enable_cameras
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `config.py` | 集中管理所有路径和默认参数（改这里一处，全生效） |
| `train.py` | 一键训练，封装官方 `train.py` |
| `evaluate.py` | 一键评估，自动定位最新 checkpoint |
| `diagnose.py` | 诊断 rollout，判断「没学到」还是「学到了但不稳」 |
| `analyze.py` | 分析训练日志，画多指标报告图（Loss/梯度/耗时等） |
| `record.py` | 录像 rollout，成功/失败各录一条 MP4（Blueprint 版环境） |
| `_common.py` | 内部工具（tee 日志、checkpoint 定位），勿直接运行 |
| `logs/` | 日志目录（时间戳命名，已被 `.gitignore` 忽略） |

## 任务与数据

- **任务**：`Isaac-Stack-Cube-Franka-IK-Rel-v0`（状态型，纯低维观测，无需 Cosmos）
- **算法**：`bc`（BC-RNN，LSTM horizon 10，配置 `bc_rnn_low_dim.json`）
- **数据**：`notebook/datasets/generated_dataset.hdf5`（Mimic 生成 100 条成功演示，robomimic 格式零转换）

## 常见用法

```bash
# 覆盖训练轮数 / 实验名
python training/train.py --epochs 1000 --name my_exp

# 手动指定 checkpoint 评估
python training/evaluate.py --checkpoint /path/to/model_epoch_400.pth

# 增加 rollout 次数拿更稳的成功率
python training/evaluate.py --num_rollouts 30
```

## 日志与 checkpoint 位置

- 本目录日志：`training/logs/{train,eval,diagnose}_<时间戳>.log`
- 训练 checkpoint：`/root/IsaacLab/logs/robomimic/<task>/<name>/<时间戳>/models/model_epoch_<N>.pth`

## 注意

- 中断训练（Ctrl+C）后 Isaac Sim native App 可能残留占显存，可用 `nvidia-smi` 检查、必要时清理进程。
- 训练数据与 checkpoint 均不入 git（已被 `.gitignore` 忽略），代码变更请正常 commit。
