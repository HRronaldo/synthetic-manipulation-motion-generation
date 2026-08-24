# AGENTS.md — notebook/ 工作流模块

> 父文档：[docs/AGENTS.md](AGENTS.md)。本文件只写 notebook/ 特有内容，不重复父级。

## 概述

Jupyter 驱动的数据生成流水线：Isaac Lab 仿真录制帧 → 本地着色编码视频 → 提交 Cosmos 生成增强视频。6 个文件，全部挂载进容器 `/workspace/isaaclab/`。

## 文件职责

| 文件 | 职责 | 关键点 |
|------|------|--------|
| `generate_dataset.ipynb` | ★ 主入口，21 cells 编排全流程 | 用户唯一需要直接操作的文件 |
| `notebook_widgets.py` | ipywidgets 控件工厂 | BSD-3 + Apache-2.0 双头注释（源自 Isaac Lab） |
| `notebook_utils.py` | 帧扫描、Warp 着色、视频编码 | 依赖容器内 `video_encoding` 模块（本地不可见） |
| `cosmos_request.py` | Cosmos API 客户端 | 纯 requests，无第三方框架 |
| `app.py` | Flask 服务端（Cosmos 节点用） | 不在 Isaac Lab 容器内运行！ |
| `stacking_prompt.toml` | 提示词模板 | `[prompt]`/`[variables]`/`[negative_prompt]` 三节 |

## 去哪找

| 任务 | 位置 |
|------|------|
| 加新下拉框变量 | `stacking_prompt.toml` 加 `[variables]` 条目 → `create_variable_dropdowns` 自动渲染 |
| 改默认生成参数 | `create_cosmos_params`（seed=42, control_weight=0.6, sigma_max=40, canny="Very Low"） |
| 改环境数/试验数上限 | `create_num_envs_input`(1-100, 默认9) / `create_num_trials_input`(1-100, 默认1) |
| 改着色光照方向 | `notebook_utils.DEFAULT_LIGHT_DIRECTION`（默认 (0,0,1) 垂直向下） |
| 改帧率 | `notebook_utils.DEFAULT_FRAMERATE`（24.0） |

## 数据流与约定

```
仿真输出 _isaaclab_out/
  └── {camera}_normals_trial_T_tile_E_step_S.png
  └── {camera}_semantic_segmentation_trial_T_tile_E_step_S.png
        │ get_env_trial_frames() 校验连续性(≥30帧)
        │ encode_video() Warp kernel 着色 → MP4 (24fps)
        ▼
Cosmos 输入视频 → process_video() POST /canny/submit
        │ app.py 后台线程 subprocess transfer.py (edge controlnet)
        ▼
_cosmos_out/ 增强视频 → create_download_link() base64 下载
```

- trial 有效性判定：帧数 ≥30 且 step 序列**完全连续**，否则跳过该 trial
- 相机选择按 `_normals_` 中缀反推相机名（`create_camera_input`）

## 反模式

- ❌ 不要在 app.py 里改 `CONTROL2WORLD_PATH` 为绝对路径之外乱动——它相对 cosmos_transfer1 仓库根
- ❌ 不要绕过 `secure_filename` 直接保存上传文件
- ❌ 不要在 notebook_utils.py 引入 CPU 重活——着色走 Warp GPU kernel，保持 device="cuda"
- ❌ jobs 字典是进程内存态——重启 app.py 即丢全部任务状态，勿当持久存储

## 注意事项

- `notebook_utils.encode_video` 导入的 `video_encoding` 模块仅存在于容器镜像内，本地 IDE 会报 ImportError——属正常现象
- `app.py` 的 `/canny/result` 下载后异步清理临时目录并删除 job 记录——结果只能下载一次
