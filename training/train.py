"""一键训练脚本。

作用：
    封装官方 robomimic 的 train.py，把「固定默认参数 + 自动生成日志文件 +
    输出同时打到终端和日志」这几件事做好，让训练变成一条命令。

    说白了：这个脚本自己不动手训练，而是「组装好官方命令并启动官方脚本」，
    具体见 main() 里 subprocess 那段和 _common.run_with_tee。
    （为什么不用 import 官方代码：官方 train.py 一加载就会拉 Isaac Sim，
     必须让它自己在独立进程里跑，见 _common.py 顶部注释。）

用法：
    python training/train.py                # 用 config 默认参数，训满 2000 epoch
    python training/train.py --epochs 1000  # 覆盖训练轮数
    python training/train.py --name my_exp  # 覆盖实验名

后台运行（推荐，训练约 1~1.5 小时）：
    nohup python training/train.py > /dev/null 2>&1 &
    tail -f training/logs/train_*.log
"""
import argparse       # 标准库：解析命令行参数（就是 --xxx 那些）
import sys            # sys.executable 取出当前 Python 解释器路径
from datetime import datetime   # 生成日志文件名的时间戳
from pathlib import Path        # 跨平台路径

# 把本文件所在目录加进 Python 搜索路径，让下面能 import 到同目录的 config / _common
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config   # noqa: E402   # 所有路径和默认参数
import _common  # noqa: E402   # 内部工具（run_with_tee 等）


def main():
    """程序入口：解析参数 → 组装官方命令 → 交给 run_with_tee 执行并记日志。"""

    # ---- 解析命令行参数 ----
    # argparse：Python 自带的命令行参数解析器。
    # 它把 "python training/train.py --epochs 1000" 里的 "--epochs 1000" 读出来，
    # 自动转成 int 并存进 args.epochs。没传的参数就用 default。
    parser = argparse.ArgumentParser(description="一键训练 BC 策略")
    # 训练轮数：数据只有 100 条，BC-RNN 要训到 2000 轮才开始见效。
    # 上次只训 426 轮就中断、效果不佳，所以留这个参数控制训练量。
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="训练轮数")
    # 实验名：会作为 checkpoint 日志目录的二级目录名（logs/robomimic/<task>/<name>/...）。
    # 换名字 = 开一个全新的实验，新旧结果互不干扰。
    parser.add_argument("--name", type=str, default=config.TRAIN_NAME, help="实验名")
    # 训练数据：Mimic 生成的那份 HDF5，默认就是 config 里配置的 generated_dataset.hdf5。
    # 想换数据（比如用 annotated_dataset.hdf5）就 --dataset 指定。
    parser.add_argument("--dataset", type=str, default=str(config.DATASET_PATH), help="训练数据路径")
    args = parser.parse_args()   # 真正执行解析，结果存进 args 对象

    # ---- 生成日志文件路径 ----
    # 每次训练一个独立日志文件，文件名带时间戳（精确到秒），不会互相覆盖。
    # 形如：training/logs/train_20260828_143000.log
    log_path = config.LOG_DIR / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"

    # ---- 组装要执行的官方命令 ----
    # cmd 是一个字符串列表，subprocess 会把它们拼起来当作一条命令行去跑。
    # 拆开看：
    #   sys.executable             → 当前 Python 解释器（conda env isaaclab 的 python）
    #   config.TRAIN_SCRIPT        → 官方 train.py 的绝对路径
    #   后面 --task / --algo ...   → 传给官方脚本的参数
    cmd = [
        sys.executable, str(config.TRAIN_SCRIPT),   # 例：python /root/IsaacLab/scripts/.../train.py
        "--task", config.TASK_NAME,                  # 任务：状态型叠方块环境
        "--algo", config.ALGO,                       # 算法：bc（行为克隆 → BC-RNN）
        "--dataset", args.dataset,                   # 数据：Mimic 生成的 HDF5
        "--name", args.name,                         # 实验名
        "--epochs", str(args.epochs),                # 训练轮数（命令参数必须是字符串，故 str()）
    ]
    # 这条 cmd 最终等价于手动在终端敲：
    #   python /root/IsaacLab/scripts/imitation_learning/robomimic/train.py \
    #     --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --algo bc \
    #     --dataset <...>/generated_dataset.hdf5 --name bc_state_franka_stack --epochs 2000

    # ---- 打印提示，然后交给 run_with_tee 执行 ----
    print(f"[train] 命令: {' '.join(cmd)}")   # 把要跑的命令打印出来，方便核对
    print(f"[train] 日志: {log_path}")         # 提示日志文件在哪，方便 tail -f 看进度
    # run_with_tee 会：启动官方 train.py 子进程 → 实时把输出打到终端 + 写进日志
    ret = _common.run_with_tee(cmd, log_path, config.ISAACLAB_ROOT)

    print(f"[train] 退出码: {ret}")
    # 退出码原样透传给 shell：0=成功，非0=失败，
    # 这样 nohup 后台跑时外部能根据它判断这次训练成没成。
    sys.exit(ret)


# Python 约定：只有当"直接运行本文件"时才执行 main()。
# 如果本文件被别的脚本 import，则不会自动跑 main()（避免副作用）。
if __name__ == "__main__":
    main()
