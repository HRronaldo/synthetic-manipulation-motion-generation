"""一键评估脚本。

作用：
    封装官方 play.py。play.py 会加载训练好的模型，在仿真环境里让机械臂
    从头跑一遍完整任务（rollout），最后打印「成功了多少次 / 共多少次」。
    这个脚本额外做了两件事：① 自动找最新 checkpoint（不用手记路径）；
    ② 实时输出 + 自动日志归档。

    一个 rollout = 从重置环境开始，机械臂按策略连续输出动作直到
    任务完成 / 失败 / 达到步数上限（horizon）的一次完整尝试。
    rollout 次数越多，成功率估计越可靠（10 次里成 2 次 → 约 20%）。

用法：
    python training/evaluate.py                       # 自动找最新 checkpoint，10 次 rollout
    python training/evaluate.py --checkpoint <路径>    # 手动指定 checkpoint
    python training/evaluate.py --num_rollouts 30     # 增加 rollout 次数拿更稳的成功率
"""
import argparse       # 解析命令行参数
import sys            # sys.executable 取当前解释器路径
from datetime import datetime   # 日志文件名时间戳
from pathlib import Path        # 跨平台路径

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让下面能 import 同目录模块
import config   # noqa: E402   # 路径与默认参数
import _common  # noqa: E402   # 工具函数


def main():
    """程序入口：解析参数 → 定位 checkpoint → 组装官方命令 → 执行并记日志。"""

    # ---- 解析命令行参数 ----
    parser = argparse.ArgumentParser(description="一键评估 BC 策略")
    # checkpoint：要加载的模型文件。不传则自动找最新的一份。
    parser.add_argument("--checkpoint", type=str, default=None, help="手动指定 checkpoint 路径（不传则自动找最新）")
    # rollout 次数：绕通越多成功率越接近真实水平。默认 10 次。
    parser.add_argument("--num_rollouts", type=int, default=config.EVAL_NUM_ROLLOUTS, help="评估 rollout 次数，越多成功率越稳")
    # 单次最大步数：演示约 253 步，留 800 步余量。
    parser.add_argument("--horizon", type=int, default=config.EVAL_HORIZON, help="单次 rollout 步数上限")
    # 随机种子：固定后结果可复现，方便对比不同 checkpoint。
    parser.add_argument("--seed", type=int, default=config.EVAL_SEED, help="随机种子")
    # 实验名：自动找 checkpoint 时用来定位目录（logs/robomimic/<task>/<name>/...）。
    parser.add_argument("--name", type=str, default=config.TRAIN_NAME, help="实验名（自动找 checkpoint 用）")
    args = parser.parse_args()

    # ---- 确定用哪个 checkpoint ----
    checkpoint = args.checkpoint
    if checkpoint is None:
        # 没手动指定 → 用 _common 去官方目录自动找最新的那个（省得记时间戳路径）
        checkpoint = _common.find_latest_checkpoint(args.name)
        if checkpoint is None:
            print("[evaluate] 未找到 checkpoint，请先训练或 --checkpoint 手动指定")
            sys.exit(1)  # 提前结束，退出码 1 表示出错
    print(f"[evaluate] checkpoint: {checkpoint}")

    # ---- 生成日志文件路径 ----
    log_path = config.LOG_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.log"

    # ---- 组装官方 play.py 命令 ----
    # 和 train.py 一样，用 subprocess 启动官方 play.py（独立进程）。
    cmd = [
        sys.executable, str(config.PLAY_SCRIPT),   # python /root/IsaacLab/scripts/.../play.py
        "--task", config.TASK_NAME,                # 任务：评估环境必须和训练环境同名
        "--checkpoint", str(checkpoint),           # 加载哪个模型（必须是 .pth 文件）
        "--num_rollouts", str(args.num_rollouts),  # 跑几次 rollout
        "--horizon", str(args.horizon),            # 单次最大步数
        "--seed", str(args.seed),                  # 随机种子
        "--headless",                              # 无界面跑（服务器没有显示器）
    ]

    # ---- 打印提示并执行 ----
    print(f"[evaluate] 命令: {' '.join(cmd)}")
    print(f"[evaluate] 日志: {log_path}")
    ret = _common.run_with_tee(cmd, log_path, config.ISAACLAB_ROOT)  # 执行 + 双写日志

    print(f"[evaluate] 退出码: {ret}")
    sys.exit(ret)


if __name__ == "__main__":
    main()
