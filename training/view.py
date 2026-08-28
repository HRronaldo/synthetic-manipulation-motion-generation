"""Livestream 实时查看脚本：在浏览器里看机械臂夹方块的实时画面。

为什么需要它？
    record.py 录的 MP4 是「事后回放」，要下载到本地才能看；
    本脚本通过 Isaac Sim 的 WebRTC Livestream，把仿真画面实时推到浏览器，
    你能亲眼看到机械臂「现在」正在做什么，还能用鼠标拖拽旋转视角。

关键机制（headless 服务器看 GUI 的正解）：
    - 服务器没有显示器，无法弹本地窗口；但 Isaac Sim 支持 --livestream，
      把 3D viewport 通过 WebRTC 流式传输，浏览器访问即可看。
    - --livestream 2 = 走「私有/本地网络」模式（默认端口 8211），
      配合 SSH 隧道把服务器端口转发到本地，浏览器打开 http://localhost:8211 就能看。
    - livestream 开启时会自动强制 headless（主机无窗口，但远端有交互式窗口）。

为什么用状态版任务（TASK_NAME）而不是 Blueprint 版？
    - 状态版没有相机 sensor，viewport 直接渲染整个场景（机械臂+方块+桌面），
      无需 --enable_cameras，最省事。
    - viewport 渲染的是整个 USD 场景，与观测用哪几个 key 无关，
      所以用状态版也能看到完整的机械臂夹方块画面。

用法（三步，缺一不可）：
    1. 本地另开一个终端，开 SSH 隧道（把服务器 8211/8899 端口转发到本地）：
       ssh -L 8211:localhost:8211 -L 8899:localhost:8899 <用户名>@<服务器IP>
    2. 服务器上跑：
       python training/view.py --livestream 2 --sleep 0.2
    3. 本地浏览器打开 http://localhost:8211 ，即可看到实时画面。
       看完按 Ctrl+C 退出脚本。

参数速查：
    --livestream {0,1,2}  开 livestream（2=私有网络，配合 SSH 隧道用）
    --sleep 秒            每步动作之间停顿多久，越大越慢越容易看清
    --max-steps 步数      跑多少步后停住保持画面（默认 200）
"""
import argparse
import sys
import time
from pathlib import Path

# 先把本文件所在目录加进搜索路径，让 import config / _common 能找到
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config   # noqa: E402   # 路径与默认参数
import _common  # noqa: E402   # 工具函数（find_latest_checkpoint）

# 本脚本和 diagnose.py/record.py 一样，在本进程真实驱动 Isaac Sim
from isaaclab.app import AppLauncher

# ---- 解析命令行参数 ----
parser = argparse.ArgumentParser(description="livestream 实时查看仿真画面")
parser.add_argument("--task", type=str, default=config.TASK_NAME)          # 状态版任务（无相机）
parser.add_argument("--checkpoint", type=str, default=None, help="手动指定 checkpoint 路径")
parser.add_argument("--sleep", type=float, default=0.2, help="每步停顿秒数，越大越慢越容易看清")
parser.add_argument("--max-steps", type=int, default=200, help="跑多少步后停住保持画面")
parser.add_argument("--disable_fabric", action="store_true", default=False)
# AppLauncher 自带一批启动参数（--headless --livestream --device 等），一次性注册进来
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ---- 启动 Isaac Sim（--livestream 2 会在这里生效，开启 WebRTC 流式传输）----
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- App 启动完成后，才能安全 import 这些 Isaac 相关模块 ----
import copy                          # 深拷贝观测字典
import gymnasium as gym              # RL 标准环境接口
import numpy as np                   # 数值计算
import random                        # 随机数
import torch                         # 深度学习框架

import robomimic.utils.file_utils as FileUtils     # robomimic：从 ckpt 加载策略
import robomimic.utils.torch_utils as TorchUtils   # robomimic：获取 torch 设备
from isaaclab_tasks.utils import parse_env_cfg     # 解析任务配置


def main():
    """加载策略 → 慢速 rollout（每步 sleep）→ 保持 livestream 运行等用户看完。"""

    # ---- 确定 checkpoint（没传则自动找最新）----
    checkpoint = args_cli.checkpoint
    if checkpoint is None:
        checkpoint = _common.find_latest_checkpoint()
        if checkpoint is None:
            print("[view] 未找到 checkpoint，请先训练或 --checkpoint 手动指定")
            return
    print(f"[view] checkpoint: {checkpoint}")

    # ---- 构造仿真环境（状态版，viewport 直接渲染场景）----
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None    # 关掉步数超时（自己控制步数）
    env_cfg.recorders = None                # 不记录数据
    env_cfg.terminations.success = None     # 不判成功（只看画面）

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # ---- 固定随机种子 ----
    torch.manual_seed(config.EVAL_SEED)
    np.random.seed(config.EVAL_SEED)
    random.seed(config.EVAL_SEED)
    env.seed(config.EVAL_SEED)

    # ---- 加载训练好的策略 ----
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=str(checkpoint), device=device)

    # ---- 慢速 rollout：每步 sleep，让浏览器里能看清机械臂每一个动作 ----
    policy.start_episode()
    obs_dict, _ = env.reset()
    print(f"[view] 环境已重置，开始慢速 rollout（每步 sleep {args_cli.sleep} 秒）...")
    print("[view] 现在就可以打开浏览器看 http://localhost:8211 了")

    for i in range(args_cli.max_steps):
        # 深拷贝观测，避免改动污染 env 内部数据
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])
        # 策略输出动作 → 执行一步
        actions = policy(obs)
        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        # 停一下，让画面在浏览器里多停留，方便看清动作
        time.sleep(args_cli.sleep)
        if terminated or truncated:
            print(f"[view] 第 {i} 步环境终止（方块掉落/步数到），画面保持中...")
            break

    # ---- rollout 结束，保持 livestream 运行，让用户继续看/拖视角 ----
    print("[view] rollout 结束，livestream 保持运行中，你可以拖动视角随意看。")
    print("[view] 看完按 Ctrl+C 退出。")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[view] 收到 Ctrl+C，退出")
    finally:
        env.close()             # 关闭环境
        simulation_app.close()  # 关闭 Isaac Sim App，释放 GPU/内存


if __name__ == "__main__":
    main()
