"""录像脚本：加载训练好的 policy 跑 rollout，把机械臂动作录成 MP4 视频。

为什么需要它？
    evaluate.py（封装官方 play.py）只打印「成功率」这个数字，看不到机械臂实际怎么动。
    本脚本在 rollout 的每一步抓取相机画面，录成 MP4，让你能亲眼看到机械臂
    是「稳准狠地叠方块」还是「乱抓一通」。成功、失败各录一条，方便对比。

关键设计点（为什么这么写）：
    1. 用 Blueprint 版任务（Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0）录像。
       因为状态版任务（Isaac-Stack-Cube-Franka-IK-Rel-v0）没有任何相机，录不了像；
       而 Blueprint 版自带两个渲染 RGB 的相机（table_cam 平视、table_high_cam 俯视）。
    2. 复用状态版训练的 checkpoint。
       两者的 policy 观测（9 个低维 key）完全一致，观测/动作维度都对得上，
       所以状态版训出来的模型直接放进 Blueprint 版环境跑，无需重新训练。
    3. 关闭 rgb_camera 观测组的「存 PNG」开关。
       否则每步会往磁盘写 4 张 PNG（2 相机 × 法线/分割），几百步就是上千张垃圾文件。
       录像所需的 RGB 帧直接从相机 sensor 读（sensor.data.output["rgb"]），
       绕开那个存文件的观测组。

术语速查：
    rollout    = 从重置环境到终态的一次完整尝试
    checkpoint = 训练存的模型快照（.pth 文件）
    policy     = 训练出来的策略网络，观测进、动作出
    fps        = 视频帧率（每秒帧数）
    RGB / BGR  = 颜色通道顺序。画面数值是 RGB（红绿蓝），cv2 写视频要 BGR（蓝绿红），
                 所以要翻转一次。

用法（--enable_cameras 必须加，否则相机不渲染会报错）：
    python training/record.py --headless --enable_cameras                           # 自动找最新 checkpoint
    python training/record.py --checkpoint <路径> --headless --enable_cameras       # 指定 checkpoint
    python training/record.py --camera table_high_cam --headless --enable_cameras   # 换俯视相机
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import _common

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="录像 rollout：成功/失败各录一条 MP4")
parser.add_argument("--task", type=str, default=config.RECORD_TASK_NAME)
parser.add_argument("--checkpoint", type=str, default=None, help="手动指定 checkpoint 路径")
parser.add_argument("--horizon", type=int, default=config.EVAL_HORIZON, help="单次 rollout 步数上限")
parser.add_argument("--seed", type=int, default=config.EVAL_SEED, help="随机种子")
parser.add_argument("--camera", type=str, default="table_cam", help="table_cam 平视 / table_high_cam 俯视")
parser.add_argument("--fps", type=int, default=config.RECORD_FPS, help="视频帧率")
parser.add_argument("--max-trials", type=int, default=config.RECORD_MAX_TRIALS,
                    help="最多跑几次 rollout 来找「成功」和「失败」各一条")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy
import gymnasium as gym
import numpy as np
import random
import torch
import cv2

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from isaaclab_tasks.utils import parse_env_cfg


def rollout(policy, env, success_term, horizon, device, camera_name):
    """跑一次 rollout，同时每步抓取相机帧。返回 (是否成功, 帧列表)。"""
    policy.start_episode()
    obs_dict, _ = env.reset()
    frames = []

    def grab():
        rgb = env.scene.sensors[camera_name].data.output["rgb"]
        return rgb[0].cpu().numpy()

    frames.append(grab())
    for i in range(horizon):
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])
        actions = policy(obs)
        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        frames.append(grab())
        if bool(success_term.func(env, **success_term.params)[0]):
            return True, frames
        elif terminated or truncated:
            return False, frames
    return False, frames


def encode_video(frames, out_path, fps):
    """把 RGB 帧列表编码成 MP4。cv2 要 BGR，所以每帧做 RGB→BGR 翻转。"""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    writer = None
    for codec in ("mp4v", "avc1"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if writer.isOpened():
            break
    if writer is None or not writer.isOpened():
        print(f"[record] 无法打开视频编码器，无法保存 {out_path}")
        return False
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return True


def main():
    checkpoint = args_cli.checkpoint
    if checkpoint is None:
        checkpoint = _common.find_latest_checkpoint()
        if checkpoint is None:
            print("[record] 未找到 checkpoint，请先训练或 --checkpoint 手动指定")
            return
    print(f"[record] checkpoint: {checkpoint}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    rgb_group = getattr(env_cfg.observations, "rgb_camera", None)
    if rgb_group is not None:
        for attr in ("table_cam_normals", "table_cam_segmentation",
                     "table_high_cam_normals", "table_high_cam_segmentation"):
            term = getattr(rgb_group, attr, None)
            if term is not None and isinstance(term.params, dict):
                term.params["save_image_to_file"] = False

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    if args_cli.camera not in env.scene.sensors:
        print(f"[record] 找不到相机 '{args_cli.camera}'，可用相机: {list(env.scene.sensors.keys())}")
        env.close()
        return

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=str(checkpoint), device=device)

    out_dir = config.VIDEO_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    success_frames = None
    failure_frames = None
    for trial in range(args_cli.max_trials):
        ok, frames = rollout(policy, env, success_term, args_cli.horizon, device, args_cli.camera)
        tag = "成功" if ok else "失败"
        print(f"[record] trial {trial}: {tag}，共 {len(frames)} 帧")
        if ok and success_frames is None:
            success_frames = frames
        if not ok and failure_frames is None:
            failure_frames = frames
        if success_frames is not None and failure_frames is not None:
            break

    if success_frames is not None:
        p = out_dir / "success.mp4"
        if encode_video(success_frames, p, args_cli.fps):
            print(f"[record] 成功视频已保存: {p}")
    if failure_frames is not None:
        p = out_dir / "failure.mp4"
        if encode_video(failure_frames, p, args_cli.fps):
            print(f"[record] 失败视频已保存: {p}")
    if success_frames is None and failure_frames is None:
        print("[record] 没跑到任何 rollout，请检查 checkpoint 是否可用")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
