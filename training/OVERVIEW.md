# training/ 工具包 — 完整梳理

> 本文档记录这套 BC 训练工具包「做了什么、为什么这么设计、踩过哪些坑」，用于封存后回顾。
> 快速上手请看 `README.md`；本文是完整复盘。

---

## 一句话概括

封装 Isaac Lab 官方 robomimic 脚本，让「训练一个机械臂叠方块的策略」从散落的命令行，变成 `python training/xxx.py` 一键搞定的可复用工具包。

---

## 一、我们完成了什么（背景与成果）

从零到一，完整经历了一次机器人模仿学习（Imitation Learning）全流程：

```
① 数据生成（notebook 跑的 Mimic 轨迹合成）
   → generated_dataset.hdf5（100 条成功演示，纯低维 state/action）
② 训练（Robomimic BC-RNN，2000 epoch）
   → checkpoint 每 100 epoch 存一份，Loss 收敛正常
③ 可视化（analyze.py）
   → 训练报告图（Loss / Log_Likelihood / 梯度范数 / 耗时）
④ 评估（evaluate.py / diagnose.py）
   → 成功率（约 9%），诊断「学会了抓、没学会叠」
⑤ 录像（record.py）
   → success.mp4 + failure.mp4 对比视频
⑥ 实时查看（view.py，livestream）
   → 浏览器里看机械臂实时动作（未验证）
```

**最终成果**：策略成功率约 9%（11 次里成功 1 次），符合「100 条演示 + BC 叠 3 方块」的预期水平（官方参考 1000 条演示才有 62%+）。

---

## 二、文件清单

| 文件 | 作用 | 关键点 |
|---|---|---|
| `config.py` | 集中配置所有路径/任务/超参 | 改一处全生效 |
| `_common.py` | 内部工具：`run_with_tee`（子进程+日志）、`find_latest_checkpoint` | 只依赖标准库，任何时机 import 都安全 |
| `train.py` | 一键训练 | subprocess 封装官方 train.py |
| `evaluate.py` | 一键评估 | subprocess 封装官方 play.py，自动找 checkpoint |
| `diagnose.py` | 诊断一次 rollout | 文字可视化机械臂行为（本进程驱动 Isaac Sim） |
| `analyze.py` | 分析训练日志画报告图 | 正则挖日志 JSON + matplotlib |
| `record.py` | 录像（成功/失败各一条 MP4） | Blueprint 版环境 + 复用 checkpoint |
| `view.py` | livestream 实时查看 | WebRTC 推到浏览器（未验证） |
| `__init__.py` | 包声明 | 纯 docstring |
| `README.md` | 快速上手 | — |
| `logs/` | 日志归档 | 已被 .gitignore 忽略 |

---

## 三、端到端工作流（命令速查）

```bash
cd /root/gpufree-data/isaac_sim_learning/synthetic-manipulation-motion-generation

# 1. 训练（后台，约 1~1.5 小时）
nohup python training/train.py > /dev/null 2>&1 &
tail -f training/logs/train_*.log

# 2. 训练报告图
python training/analyze.py

# 3. 评估成功率
python training/evaluate.py

# 4. 诊断（文字看机械臂怎么动）
python training/diagnose.py --headless

# 5. 录像（成功/失败各一条）
python training/record.py --headless --enable_cameras

# 6. 实时查看（先本地开 SSH 隧道，再跑，浏览器打开 localhost:8211）
ssh -L 8211:localhost:8211 -L 8899:localhost:8899 <用户名>@<服务器IP>
python training/view.py --livestream 2 --sleep 0.2
```

---

## 四、关键设计决策（为什么这么写）

### 1. 任务选「状态版」而非「视觉版」
- 训练/评估用 `Isaac-Stack-Cube-Franka-IK-Rel-v0`（状态版）：纯低维观测（9 个 key），**没有相机**。
- 与 Mimic 生成的数据格式（`data/demo_0/obs/...`）零转换直接训练。
- 视觉版（Visuomotor）需要图像观测 + Cosmos 增强，链路长、且我们没有 H100。

### 2. train/evaluate 用 subprocess 封装官方脚本，不用 import
- 官方 `train.py`/`play.py` 一被 import 就在模块顶层拉起 Isaac Sim 整个 App。
- 用 `subprocess.Popen` 开**独立进程**跑官方脚本，等价于手动敲命令，风险隔离、生命周期官方自管。
- `sys.executable` = 当前解释器（conda env 的 python），让官方脚本复用同环境所有库。

### 3. checkpoint 自动查找
- 官方 checkpoint 结构：`logs/robomimic/<task>/<name>/<时间戳>/models/model_epoch_<N>.pth`。
- `find_latest_checkpoint` 自动找「最新一次训练的最新 checkpoint」，免手记时间戳路径。

### 4. 录像用「Blueprint 版环境」+ 复用状态版 checkpoint
- 状态版任务**没有任何相机**，录不了像。
- Blueprint 版（`Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0`）自带两个渲染 RGB 的相机（table_cam 平视 / table_high_cam 俯视，704×1280）。
- 关键：两者的 **policy 观测（9 个低维 key）完全一致**，所以状态版训出的 checkpoint 可直接在 Blueprint 版环境跑，无需重新训练。
- 录像时关闭 rgb_camera 观测组的「存 PNG」开关（否则每步落 4 张垃圾 PNG），RGB 帧直接从 `env.scene.sensors["table_cam"].data.output["rgb"]` 读。

### 5. 中文字体自动检测（analyze.py）
- matplotlib 默认无中文字体，中文渲染成方框。
- 用 `font_manager` 探测候选字体（Noto Sans CJK SC 等），命中则配置，找不到则提示安装 `fonts-noto-cjk`。

---

## 五、踩过的坑（经验沉淀）

| # | 坑 | 根因 | 修复 |
|---|---|---|---|
| 1 | checkpoint 找不到 | `find_latest_checkpoint` 的 glob 少了 `models/` 前缀（文件实际在 `models/` 子目录） | `glob("model_epoch_*.pth")` → `glob("models/model_epoch_*.pth")` |
| 2 | 录像报「camera spawned without --enable_cameras」 | headless 模式相机默认不渲染，需显式开 | 命令行加 `--enable_cameras` |
| 3 | 训练卡在 overwrite 提示 | 官方 train.py 目录已存在时 `input()` 问 y/n，子进程 stdin 未接管 | `run_with_tee` 里 `stdin=PIPE` + 自动喂 `y` |
| 4 | 报告图中文方框 | matplotlib 无中文字体 | 字体自动检测 + 配置 |
| 5 | 训练 GPU 占用仅 4% | 小模型 + 数据加载瓶颈，不是算力瓶颈 | 正常现象，勿误判；提速方向是减 epoch/增 batch |
| 6 | 成功率 0% | 100 条数据 + BC 叠 3 方块（4 段序列任务）太难 | 正常现象；要提升需加数据或换简单任务 |

**教训**：`--enable_cameras` 是「相机 sensor 离屏渲染」开关，与 viewport/livestream 是两回事；viewport 渲染整个 USD 场景不需要它，但环境里有相机 sensor 就必须开。

---

## 六、数据 / 任务 / 模型约定

- **训练数据**：`notebook/datasets/generated_dataset.hdf5`（Mimic 生成 100 条成功演示，robomimic 格式零转换）
- **训练/评估任务**：`Isaac-Stack-Cube-Franka-IK-Rel-v0`（状态版，纯低维）
- **录像任务**：`Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0`（带 RGB 相机）
- **算法**：`bc`（BC-RNN，LSTM horizon 10，GMM 5 modes，配置 `bc_rnn_low_dim.json`）
- **policy 观测**：9 个低维 key，策略只吃其中 4 个（eef_pos/eef_quat/gripper_pos/object）
- **动作**：7 维（6 维 EEF 相对位移 IK-Rel + 1 维夹爪）
- **checkpoint 位置**：`/root/IsaacLab/logs/robomimic/<task>/<name>/<时间戳>/models/model_epoch_<N>.pth`
