"""诊断脚本：加载 policy 跑一次 rollout，文字可视化机械臂行为。

为什么需要它？
    评估只告诉你「成功/失败」，不告诉你「为什么失败」。
    这个脚本在机械臂每次行动的关键瞬间，把关键状态量打印成一排排文字，
    让你能"看到"机械臂在想什么、做对了哪一步、在哪一步翻车。

    用于区分「没学到」vs「学到了但不稳」：
      - act_dxyz 幅度是否合理（IK 相对模式应 ±0.1 量级；动辄 ±1 说明尺度错乱）
      - dist2cube 是否下降（机械臂有没有在靠近方块）
      - cubes_z 是否升高（方块有没有被举起来）
      - 结束原因：成功 / 方块掉落（cube dropping）/ 跑满 horizon

术语：
    rollout    = 从重置环境开始，机械臂连续决策直到终态的一次完整尝试
    policy     = 训练出来的策略网络，输入观测、输出动作
    obs        = 观测（状态），这里是一堆低维数值（末端位置、夹爪、方块位置等）
    eef        = 末端执行器（end effector），即夹爪
    horizon    = 单次 rollout 最多跑多少步

用法：
    python training/diagnose.py --headless                       # 自动找最新 checkpoint
    python training/diagnose.py --checkpoint <路径> --seed 101 --headless
"""
import argparse
import sys
from pathlib import Path

# 先把本文件所在目录加进搜索路径，让下面的 import config / _common 能找到
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config   # noqa: E402   # 路径与默认参数
import _common  # noqa: E402   # 工具函数（find_latest_checkpoint）

# 注意：本脚本和 train/evaluate 不同，它要"在本进程里真实驱动 Isaac Sim 仿真"，
# 所以会直接 import AppLauncher 并启动，时序见下面的注释。
from isaaclab.app import AppLauncher

# ---- 解析命令行参数 ----
parser = argparse.ArgumentParser(description="诊断 rollout：文字可视化机械臂行为")
parser.add_argument("--task", type=str, default=config.TASK_NAME)          # 任务名
parser.add_argument("--checkpoint", type=str, default=None, help="手动指定 checkpoint 路径")  # 模型
parser.add_argument("--horizon", type=int, default=400, help="单次 rollout 步数上限")          # 最大步数
parser.add_argument("--seed", type=int, default=config.EVAL_SEED, help="随机种子")             # 种子
parser.add_argument("--disable_fabric", action="store_true", default=False)                    # 关掉数据管道优化（调试用）
# AppLauncher 自带一批启动参数（如 --headless --device 等），一次性注册进来
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ---- 启动 Isaac Sim ----
# AppLauncher(...) 构造 = 真正把 Isaac Sim 原生 App 拉起来。
# 必须在这一步之后，才能 import isaaclab 的其余模块
# （否则会触发 "XXX loaded before SimulationApp was started" 的警告/错误）。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 上面 App 启动完成后，才能安全地 import 这些 Isaac 相关模块 ----
import copy                          # 深拷贝观测字典（避免改坏 env 内部数据）
import gymnasium as gym              # RL 标准环境接口
import numpy as np                   # 数值计算
import random                        # 随机数
import torch                         # 深度学习框架
import robomimic.utils.file_utils as FileUtils     # robomimic：从 ckpt 加载策略
import robomimic.utils.torch_utils as TorchUtils   # robomimic：获取 torch 设备
from isaaclab_tasks.utils import parse_env_cfg     # 解析任务配置


def main():
    """主流程：构造环境 → 加载策略 → 跑一次 rollout → 打印关键状态。"""

    # ---- 确定 checkpoint（没传则自动找最新）----
    checkpoint = args_cli.checkpoint
    if checkpoint is None:
        checkpoint = _common.find_latest_checkpoint()
        if checkpoint is None:
            print("[diagnose] 未找到 checkpoint，请先训练或 --checkpoint 手动指定")
            return
    print(f"[diagnose] checkpoint: {checkpoint}")

    # ---- 构造仿真环境配置 ----
    # parse_env_cfg：从任务名加载环境配置。device 决定用 CPU 还是 GPU 推理。
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,                        # 只开 1 个并行环境（诊断不需要并行多跑）
        use_fabric=not args_cli.disable_fabric,
    )
    # 别把 9 个观测字段拼成一个向量，保持"字典"形式，方便按 key 单独取数值打印
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None      # 关掉"步数超时"这个终止条件（我们自己控制步数）
    env_cfg.recorders = None                  # 不录像、不记录数据
    # 把"成功"判定单独摘出来存到 success_term，方便自己判断何时算成功；
    # 同时从终止条件里去掉它，避免环境自己提前结束（我们要完整观察行为过程）
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    # 用配置创建环境；.unwrapped 拿到底层裸环境（剥掉 gym 的一层层包装）
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # ---- 固定随机种子，保证结果可复现 ----
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    # ---- 加载训练好的策略 ----
    # get_torch_device(try_to_use_cuda=True)：优先用 CUDA(显卡)，不行就退到 CPU
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    # policy_from_checkpoint：从 .pth 文件还原出神经网络策略（参数+结构都在文件里）
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=str(checkpoint), device=device)

    # ---- 开始一次 rollout ----
    # start_episode()：让策略进入"新回合"状态（BC-RNN 有内部记忆，要先清零）
    policy.start_episode()
    # env.reset()：把机械臂和方块摆回初始布局，返回初始观测
    obs_dict, _ = env.reset()
    # obs_dict["policy"] = 给策略用的那一组低维观测（是个 dict，key 是 9 个字段名）

    # 辅助函数：算末端到最近方块的距离，用来判断机械臂是否在靠近目标
    def eef_cube_dist(pol):
        eef = pol["eef_pos"].cpu().numpy().flatten()        # 末端位置 (x,y,z)
        cubes = pol["cube_positions"].cpu().numpy().flatten().reshape(3, 3)  # 3 个方块，每行 (x,y,z)
        return float(np.min(np.linalg.norm(cubes - eef, axis=1)))  # 取 3 个距离的最小值

    # 辅助函数：把关键状态打印成一行文字，一眼看清当前这个小瞬间发生了什么
    def show(tag, pol, action=None):
        eef = pol["eef_pos"].cpu().numpy().flatten()      # 末端位置 (x,y,z)
        grip = pol["gripper_pos"].cpu().numpy().flatten() # 夹爪开合度（2 个手指）
        cubes = pol["cube_positions"].cpu().numpy().flatten()  # 9 个数：3 方块 × xyz
        line = f"[{tag}] eef=({eef[0]:.3f},{eef[1]:.3f},{eef[2]:.3f}) "          # 末端位置
        line += f"grip=({grip[0]:.3f},{grip[1]:.3f}) "                            # 夹爪开合
        line += f"dist2cube={eef_cube_dist(pol):.3f} "                            # 离最近方块多远
        line += f"cubes_z=({cubes[2]:.3f},{cubes[5]:.3f},{cubes[8]:.3f})"         # 3 个方块的高度 z
        if action is not None:
            a = action.cpu().numpy().flatten()            # 策略输出的动作（共 7 维）
            line += f" | act_dxyz=({a[0]:.3f},{a[1]:.3f},{a[2]:.3f}) "  # 末端位移（IK 相对模式）
            line += f"act_grip={a[6]:.3f}"                # 第 7 维 = 夹爪开合命令
        print(line)

    # 打印初始状态（机械臂在哪、方块多高、离方块多远）
    show("step 000 init", obs_dict["policy"])

    # ---- rollout 主循环 ----
    for i in range(args_cli.horizon):
        # 深拷贝当前观测，避免改动污染 env 内部的数据
        obs = copy.deepcopy(obs_dict["policy"])
        # 把每个观测字段去掉多余的维度（gym 返回带 batch 维度，压平成策略要的形状）
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])
        # 让策略基于当前观测输出动作（policy 是个可调用对象：观测进、动作出）
        actions = policy(obs)
        # robomimic 的动作是 numpy 数组，这里转成 torch、放到设备上，
        # 并重塑成 (1, 动作维度) 以匹配 env.step 期望的输入形状
        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        # env.step(actions)：在仿真里执行这一步动作、把物理世界推进一步，
        # 返回：新观测、奖励、是否终止(terminated)、是否截断(truncated)、额外信息
        obs_dict, _, terminated, truncated, _ = env.step(actions)

        # 每 50 步打印一次关键状态（step 050/100/150...），方便看到行为轨迹
        if i % 50 == 49:
            show(f"step {i+1:03d}", obs_dict["policy"], actions)

        # ---- 判断这回合是否结束 ----
        # 手动调用"成功"终止条件的判定函数：三个方块是否都按要求叠上了
        if bool(success_term.func(env, **success_term.params)[0]):
            print(f">>> 成功！第 {i+1} 步完成任务")
            break
        # 环境自己判定终止（比如方块掉出边界）或截断（步数上限）
        elif terminated or truncated:
            print(f">>> 环境终止于第 {i+1} 步 (terminated={terminated}, truncated={truncated})")
            break

    print(f"共跑 {i+1} 步")
    env.close()   # 关闭环境，释放仿真资源


if __name__ == "__main__":
    main()
    simulation_app.close()   # 关闭 Isaac Sim App，彻底释放 GPU/内存资源
