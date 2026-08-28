"""训练相关集中配置。

为什么把参数集中到一起？
    训练/评估/诊断三个脚本要用的路径、任务名、默认超参数全都写在这里，
    改一处、三个脚本一起生效。这样不用在每个脚本里到处找参数，
    也方便事后回顾"当时用的什么配置"。想换数据、换任务、改轮数，都来这里改。

新手概念速查：
    task    = 一个具体的仿真任务（本目录是「Franka 机械臂把 3 个方块叠成塔」）。
              这个任务名唯一对应一套环境配置，训练和评估必须用同一个 task 才对得上。
    algo    = 算法名。这里用 bc（Behavior Cloning 行为克隆）——直接"模仿"演示数据，
              见不到任何奖励信号，属于最简单的模仿学习。
    epoch   = 训练把整个数据集从头到尾过一遍叫 1 个 epoch。数据少时要多过几遍
              模型才够熟，所以默认 2000 轮（对 100 条数据属正常量级）。
    rollout = 评估/运行时的一次完整尝试（从重置环境到终态）。
    checkpoint = 训练过程中定期存下来的模型快照（.pth 文件），用它可以继续训或部署。
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
# 项目根目录：本文件位于 <project_root>/training/config.py，上溯一级即根目录。
# 用 __file__ 动态推导，不管项目拷到哪都能找到，不写死绝对路径。
# 在服务器上会自动解析为
# /root/gpufree-data/isaac_sim_learning/synthetic-manipulation-motion-generation
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Isaac Lab 源码根目录（editable 安装在这），官方 train.py / play.py 都在里面。
# 如果服务器上装的位置不同，改这一行即可。
ISAACLAB_ROOT = Path("/root/IsaacLab")

# 训练数据：Mimic 生成的那份 robomimic 格式 HDF5，100 条成功演示、纯低维 state/action。
DATASET_PATH = PROJECT_ROOT / "notebook" / "datasets" / "generated_dataset.hdf5"

# 本模块自己的日志子目录（训练/评估/诊断日志按时间戳归档在这里）。
# 它叫 logs 且被 .gitignore 忽略，所以日志不会污染 git。
LOG_DIR = Path(__file__).resolve().parent / "logs"

# ---------------------------------------------------------------------------
# 任务与算法
# ---------------------------------------------------------------------------
# 状态型训练/评估环境：纯低维观测（无图像），与 generated_dataset 里的观测字段完全一致，
# 所以这份数据能零转换直接训练。（如果换成 Visuomotor 那种带相机图像的任务就对不上了）
TASK_NAME = "Isaac-Stack-Cube-Franka-IK-Rel-v0"

# 算法名。官方用 algo 拼出配置钥匙 robomimic_bc_cfg_entry_point → bc_rnn_low_dim.json，
# 所以这里必须写 bc（不是 bc_rnn）。这个 json 决定了模型结构、观测选择等。
ALGO = "bc"

# ---------------------------------------------------------------------------
# 训练默认参数
# ---------------------------------------------------------------------------
# 实验名：作为 checkpoint 日志目录的二级目录名
#   logs/robomimic/<task>/<name>/<时间戳>/models/...
# 想开一个全新实验就改这里（或命令行 --name），互不干扰。
TRAIN_NAME = "bc_state_franka_stack"
# 训练轮数：数据只有 100 条，BC-RNN 要训满 2000 轮才开始表现稳定。
# （上次只训 426 轮就被中断，出现"光会抓、不会叠"的现象。）
EPOCHS = 2000

# ---------------------------------------------------------------------------
# 评估默认参数
# ---------------------------------------------------------------------------
# 单次 rollout 步数上限：人类演示约 253 步，留 800 步余量足够机械臂慢慢完成任务。
EVAL_HORIZON = 800
# 评估 rollout 次数：次数越多，成功率（成功次数/总次数）越接近真实水平。
EVAL_NUM_ROLLOUTS = 10
# 随机种子：固定后每次随机初始化（方块位置/关节初值）都一样，
# 结果可复现，方便对比不同 checkpoint 到底谁更强。
EVAL_SEED = 101

# ---------------------------------------------------------------------------
# 录像默认参数
# ---------------------------------------------------------------------------
# 录像用任务：Blueprint 版自带渲染 RGB 的相机（table_cam 平视 / table_high_cam 俯视），
# 而状态版任务（TASK_NAME）没有任何相机，录不了像。
# 关键：Blueprint 版的 policy 观测（9 个低维 key）与状态版完全一致，
# 所以状态版训练的 checkpoint 可以直接在 Blueprint 版环境里跑，无需重新训练。
RECORD_TASK_NAME = "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0"
# 录像输出目录：放在 logs 下（logs 已被 .gitignore 忽略），视频不污染 git。
VIDEO_OUTPUT_DIR = LOG_DIR / "videos"
# 录像默认参数
RECORD_FPS = 24          # 视频帧率（每秒帧数），与数据生成时一致
RECORD_MAX_TRIALS = 10   # 最多跑几次 rollout 来找「成功」和「失败」各一条

# ---------------------------------------------------------------------------
# 官方脚本路径（一般不用改）
# ---------------------------------------------------------------------------
ROBOMIMIC_DIR = ISAACLAB_ROOT / "scripts" / "imitation_learning" / "robomimic"
TRAIN_SCRIPT = ROBOMIMIC_DIR / "train.py"   # 官方训练脚本
PLAY_SCRIPT = ROBOMIMIC_DIR / "play.py"     # 官方评估脚本
