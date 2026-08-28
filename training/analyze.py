"""训练日志分析脚本：从训练日志里挖出所有关键指标并画成一张报告图。

作用：
    训练时每个 epoch 结束都会打印一段 JSON，里面除了 Loss，还有
    Log_Likelihood / Policy_Grad_Norms / Time_Epoch / Time_Data_Loading 等指标。
    这些 JSON 已被 train.py 的 run_with_tee 完整写进 training/logs/train_*.log。
    本脚本把这些 JSON 提取出来，每个指标画一张子图，拼成一张「训练报告图」。

用法：
    python training/analyze.py                # 自动分析最新一份训练日志
    python training/analyze.py --log <路径>    # 指定某一份日志
    python training/analyze.py --smth 20      # 平滑窗口改成 20（看整体趋势）

依赖：matplotlib（Isaac Lab 环境已自带）。
中文字体：若图中中文显示成方框（豆腐块），说明系统缺中文字体，
    服务器上执行 `sudo apt-get install -y fonts-noto-cjk` 后重跑即可。
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# 匹配 "Train Epoch N\n{...JSON...}" 块
EPOCH_RE = re.compile(r"Train Epoch (\d+)\n(\{.*?\n\})", re.DOTALL)

# 要画的指标：(JSON 字段名, 子图标题, y 轴标签)。缺失字段自动跳过。
METRICS = [
    ("Loss",              "Loss（越小越好）",            "损失，越小越好"),
    ("Log_Likelihood",    "Log_Likelihood（越大越好）",  "对数似然，越大越好"),
    ("Policy_Grad_Norms", "Policy_Grad_Norms（梯度范数）", "梯度范数（越稳越好）"),
    ("Time_Epoch",        "Time_Epoch（秒/epoch）",      "每个 epoch 耗时（秒）"),
    ("Time_Data_Loading", "Time_Data_Loading（秒）",      "数据加载耗时（秒）"),
]

# 中文字体候选（按优先级从上到下探测）
_CJK_FONTS = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
    "Source Han Sans SC", "Source Han Sans CN",
    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
    "SimHei", "Microsoft YaHei", "PingFang SC",
    "AR PL UMing CN", "AR PL UKai CN",
]


def setup_chinese_font():
    """配置 matplotlib 使用中文字体，返回找到的字体名（没找到返回 None）。

    找不到中文字体时，中文会渲染成方框（豆腐块），此时打印安装提示。
    """
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False  # 让负号正常显示
            return name
    return None


def find_latest_log():
    logs = sorted(config.LOG_DIR.glob("train_*.log"))
    return logs[-1] if logs else None


def parse_log(log_path):
    text = log_path.read_text(encoding="utf-8")
    result = []
    for m in EPOCH_RE.finditer(text):
        epoch = int(m.group(1))
        try:
            metrics = json.loads(m.group(2))
        except json.JSONDecodeError:
            continue
        result.append((epoch, metrics))
    return result


def smooth(values, window):
    if window <= 1:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def main():
    parser = argparse.ArgumentParser(description="分析训练日志，画训练报告图")
    parser.add_argument("--log", type=str, default=None,
                        help="日志文件路径（默认自动找最新的 train_*.log）")
    parser.add_argument("--smth", type=int, default=10,
                        help="平滑窗口大小，1 表示不平滑")
    args = parser.parse_args()

    font = setup_chinese_font()
    if font is None:
        print("⚠ 未找到中文字体，图中中文会显示为方框。")
        print("  服务器上执行: sudo apt-get install -y fonts-noto-cjk  然后重跑本脚本。")

    log_path = Path(args.log) if args.log else find_latest_log()
    if log_path is None or not log_path.exists():
        print("找不到训练日志，请先用 train.py 训练，或用 --log 指定路径")
        sys.exit(1)

    data = parse_log(log_path)
    if not data:
        print("日志里没解析到任何 epoch 数据")
        sys.exit(1)

    epochs = [e for e, _ in data]

    available = []
    for field, title, ylabel in METRICS:
        if field not in data[0][1]:
            continue
        values = [m[field] for _, m in data]
        available.append((title, ylabel, values))

    print(f"日志: {log_path.name}")
    print(f"解析到 {len(data)} 个 epoch（epoch {epochs[0]} ~ {epochs[-1]}）")
    if font:
        print(f"使用中文字体: {font}")
    for title, _, values in available:
        print(f"{title:28s}  {values[0]:.4f}  ->  {values[-1]:.4f}")

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (title, ylabel, values) in zip(axes, available):
        ax.plot(epochs, values, alpha=0.3, linewidth=0.8, label="原始")
        ax.plot(epochs, smooth(values, args.smth), linewidth=2,
                label=f"平滑(窗口 {args.smth})")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Epoch")
    fig.suptitle(f"训练报告 — {log_path.name}", fontsize=14)
    fig.tight_layout()

    out_path = config.LOG_DIR / f"training_report_{log_path.stem}.png"
    fig.savefig(out_path, dpi=150)
    print(f"报告图已保存: {out_path}")


if __name__ == "__main__":
    main()
