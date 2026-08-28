"""内部工具函数：命令 tee 执行、checkpoint 定位。

为什么单独放一个 _common.py？
    train.py / evaluate.py / diagnose.py 都要用到「把命令输出同时打到终端和日志」
    和「自动找最新 checkpoint」这两个功能。如果把这段逻辑在每个脚本里各复制一份，
    将来要改就要改三处、容易漏。所以抽到 _common.py 里统一维护。

为什么这个文件只依赖标准库 + config、不依赖 isaaclab 等大库？
    因为 evaluate/diagnose 等脚本里有"必须先启动 Isaac Sim 再 import 其他模块"的
    严格顺序要求，而这个工具模块是"纯逻辑"，不碰 Isaac Sim，
    所以在任何时机 import 它都安全，不会被 App 启动顺序问题连累。

命名惯例：文件名的下划线前缀 _common.py 表示「这是内部实现细节，
使用者不要直接运行它」，只作为其他脚本的公共依赖。
"""
import re            # 正则表达式库，这里用来剥离终端输出的 ANSI 颜色码
import subprocess    # 标准库：启动一个子进程执行外部命令（见 run_with_tee）
import sys           # 标准库：sys.stdout 指向终端、sys.executable 是当前解释器
from datetime import datetime   # 生成日志文件名用的时间戳
from pathlib import Path        # 跨平台路径处理

# 把本文件所在目录（training/）加进 Python 搜索路径，
# 这样下面 `import config` 才能找到同目录的 config.py。
# 因为脚本可能被 `python training/xxx.py` 或 `python -m training.xxx`
# 等不同方式启动，cwd 不固定，所以显式把文件所在目录加进 sys.path 最稳。
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402  # E402 是 lint 提示「import 不在文件顶部」，这里故意忽略

# 匹配 ANSI 颜色码的正则。例如 "\x1b[31m" 表示「开始用红色」。
# subprocess 捕获到的终端输出里常夹带这些颜色码，写在日志里很乱，
# 所以写日志前用这个正则把它们剥掉；终端显示则保留（好看）。
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def run_with_tee(cmd, log_path, cwd):
    """运行一条命令，同时把输出打到终端和一份日志文件。

    术语解释：tee 是 Linux 的一个命令，能把数据「一份给屏幕、一份给文件」，
    这里借用这个思路取名为 run_with_tee。

    自动确认覆盖：官方 train.py 在模型目录已存在时会 input() 问 "overwrite? (y/n)"，
    本函数会自动喂 'y' 确认（即覆盖旧结果）。想保留旧结果就换一个 --name。

    参数：
        cmd      (list[str])  要执行的完整命令行，例如
                              ["/opt/conda/envs/isaaclab/bin/python",
                               "/root/IsaacLab/scripts/.../train.py",
                               "--task", "Isaac-Stack-Cube-Franka-IK-Rel-v0", ...]
        log_path (Path)       日志文件的完整路径，输出会被写入到这里
                              （文件名形如 train_20260828_143000.log）
        cwd      (Path)       子进程的工作目录。官方 train.py 需要在
                              IsaacLab 根目录下运行，所以调用方传 config.ISAACLAB_ROOT

    返回：
        int     子进程的退出码（0 表示成功，非 0 表示异常）。
                这样调用方（train.py / evaluate.py）能根据退出码判断成败。
    """
    # 确保日志文件所在的 logs/ 目录存在（不存在就创建，exist_ok=True 表示已存在也不报错）
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 以"写"模式打开日志文件；encoding="utf-8" 保证中文不乱码
    with open(log_path, "w", encoding="utf-8") as f:
        # 先写两行"头信息"：完整命令 + 当前时间，方便事后回看这是一次什么运行
        f.write(f"$ {' '.join(cmd)}\n")
        f.write(f"# {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.flush()  # 强制把缓冲区内容写入磁盘，避免进程被杀时丢头信息

        # ---- 启动子进程（关键！）----
        # subprocess.Popen 会开一个全新的操作系统进程去执行 cmd。
        # 这和我们手动敲那行命令完全等价。官方 train.py 就在新进程里跑，
        # 隔离在自己进程世界里，不会污染我们脚本的状态。
        proc = subprocess.Popen(
            cmd,                            # 要执行的命令（list 形式）
            cwd=str(cwd),                   # 子进程的工作目录
            stdout=subprocess.PIPE,         # 捕获子进程的标准输出
            stderr=subprocess.STDOUT,       # 把标准错误也合并进标准输出（一行不漏）
            stdin=subprocess.PIPE,          # 接管标准输入：官方 train.py 在模型目录已存在时会
                                            #   input() 问 "overwrite? (y/n)"，必须喂 'y'，否则
                                            #   前台会卡住等输入、nohup 后台会读到 /dev/null 直接失败
            text=True,                      # 以文本模式读取（True 等价于 universal_newlines）
            bufsize=1,                      # 行缓冲：每读到一行立即同步，保证实时
        )

        # 预先写入 'y' 让官方脚本的 input() 自动读到"确认覆盖"，然后关闭 stdin。
        # 这样前台不卡、后台不崩。若子进程从不读 stdin，这几字节被丢弃，无害。
        proc.stdin.write("y\n")
        proc.stdin.flush()
        proc.stdin.close()

        # ---- 逐行读取子进程输出，双写：终端 + 日志 ----
        # proc.stdout 是一个文件对象，for line in 会一行一行吐出子进程的输出。
        for line in proc.stdout:
            sys.stdout.write(line)          # 送给终端（保留 ANSI 颜色，看进度好看）
            f.write(ANSI_RE.sub("", line))  # 写进日志（剥掉颜色码，事后 grep 方便）
            f.flush()                       # 每行都刷盘，保证日志实时、断电不丢

        # 等子进程跑完（阻塞直到它结束）
        proc.wait()
        # 返回退出码：0=成功，非0=报错
        return proc.returncode


def find_latest_checkpoint(name=None):
    """在官方输出的日志目录树里，定位「最新一次训练的最新 checkpoint」。

    官方 train.py 保存模型的目录结构是：
        <ISAACLAB_ROOT>/logs/robomimic/<task>/<name>/<时间戳>/models/model_epoch_<N>.pth
        例如 /root/IsaacLab/logs/robomimic/Isaac-Stack-Cube-Franka-IK-Rel-v0/
              bc_state_franka_stack/20260828085116/models/model_epoch_400.pth
      - <时间戳> 是每次训练开始时生成的（形如 20260828085116），
        所以同一次训练的所有文件都在同一个时间戳目录下。
      - model_epoch_<N>.pth 是第 N 个 epoch 结束时存的模型，
        save.every_n_epochs=100 决定了每 100 个 epoch 存一份。

    为什么要这个函数？
        因为那个时间戳目录名（如 20260828085116）每次跑都不一样、很难记，
        评估前还得手动去翻目录。这个函数自动把「最新那份」找出来，免去手写路径。

    参数：
        name (str, 可选) 实验名。用于拼搜索目录里 <name> 那一段。
                         不传则用 config.TRAIN_NAME 默认值。

    返回：
        Path | None  最新的 model_epoch_*.pth 文件的路径；
                     找不到（还没训练过 / 目录里没有模型）则返回 None。
    """
    # 用传入的 name 或默认的 config.TRAIN_NAME 拼出搜索起始目录
    name = name or config.TRAIN_NAME
    search_root = config.ISAACLAB_ROOT / "logs" / "robomimic" / config.TASK_NAME / name

    # 目录不存在说明还没训练过，返回 None 让调用方提示用户
    if not search_root.exists():
        return None

    # 列出该目录下所有子目录（每个子目录是一次训练 run，名字是时间戳）
    run_dirs = [d for d in search_root.iterdir() if d.is_dir()]
    if not run_dirs:
        return None

    # 取"最新一次" run：目录名是时间戳字符串（如 20260828085116），
    # 字符串按字典序排就是时间顺序，所以 max() 就拿到最新的那次训练。
    latest_run = max(run_dirs, key=lambda d: d.name)

    # 在这最新一次训练的 models/ 目录下，列出所有 checkpoint 文件
    models = list(latest_run.glob("models/model_epoch_*.pth"))
    if not models:
        return None

    # 从文件名 "model_epoch_400.pth" 里提取 epoch 号 400：
    # 先按 "_" 切分取最后一段 "400.pth"，再按 "." 切分取 "400"，转成 int。
    def epoch_num(p):
        return int(p.name.split("_")[-1].split(".")[0])

    # 取 epoch 号最大的那个 = 训练到最晚的 checkpoint
    return max(models, key=epoch_num)
