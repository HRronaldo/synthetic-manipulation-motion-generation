# 项目知识库（AGENTS.md 主文档）

**生成时间：** 2026-08-21
**提交：** be78013
**分支：** main

## 概述

NVIDIA Omniverse Blueprint：机器人合成操作运动生成（Synthetic Manipulation Motion Generation）。从少量人类演示出发，基于 Isaac Sim/Isaac Lab 仿真 + NVIDIA Cosmos 视频生成模型，批量产出机器人操作的合成运动轨迹数据集。核心代码为 Jupyter 工作流（`notebook/`），通过 Docker 容器部署运行。

## 目录结构

```
./
├── docker-compose.yml          # 单服务 isaac-lab；逐文件挂载 notebook/ 到容器 /workspace/isaaclab/
├── launch.sh                   # 容器入口点（注意：被 DLP 加密，见"注意事项"）
├── README.md                   # 官方部署说明（Ubuntu 22.04 + Docker + NVIDIA GPU）
├── notebook/                   # 核心工作流（详见 docs/AGENTS-notebook.md）
│   ├── generate_dataset.ipynb  # ★ 主入口：全部流程在此 Notebook 中编排
│   ├── notebook_widgets.py     # ipywidgets UI 控件
│   ├── notebook_utils.py       # 帧扫描/Warp 着色/视频编码工具
│   ├── cosmos_request.py       # Cosmos REST API 客户端
│   ├── app.py                  # Flask 服务，部署在 Cosmos H100 节点
│   └── stacking_prompt.toml    # Cosmos 提示词模板（变量化）
├── samples/
│   └── annotated_dataset.hdf5  # 示例 Mimic 标注数据集（挂载为容器内 datasets/）
└── docs/                       # 本知识库文档目录
```

## 去哪找

| 任务 | 位置 | 备注 |
|------|------|------|
| 理解端到端流程 | `notebook/generate_dataset.ipynb` | 21 个 cell：仿真 → 数据生成 → Cosmos 增强 |
| 改 UI 参数控件 | `notebook/notebook_widgets.py` | 全部 ipywidgets 工厂函数 |
| 改帧处理/编码逻辑 | `notebook/notebook_utils.py` | Warp GPU kernel 在此 |
| 改 Cosmos 调用参数 | `notebook/cosmos_request.py` + `notebook/app.py` | 客户端与服务端各一半 |
| 改提示词模板 | `notebook/stacking_prompt.toml` | `{变量}` 占位符由下拉框填充 |
| 改容器/挂载/GPU 配置 | `docker-compose.yml` | 镜像 `nvcr.io/nvidia/gr00t-smmg-bp:1.0` |

## 代码地图

（项目 <10 个源码文件，已全量通读；codegraph 未索引，中心度按调用关系人工判定）

| 符号 | 类型 | 位置 | 角色 |
|------|------|------|------|
| `process_video` | 函数 | notebook/cosmos_request.py | 提交视频→轮询→下载结果（submit/status/result 三段式） |
| `test_connection` | 函数 | notebook/cosmos_request.py | TCP 连通性预检 |
| `PromptManager` | 类 | notebook/notebook_widgets.py | 用 TOML 模板+下拉框值实时拼装提示词 |
| `create_variable_dropdowns` | 函数 | notebook/notebook_widgets.py | 从 stacking_prompt.toml 渲染变量下拉框 |
| `create_cosmos_params` | 函数 | notebook/notebook_widgets.py | seed/control_weight/sigma_max/canny_strength 控件 |
| `encode_video` | 函数 | notebook/notebook_utils.py | 帧 PNG 序列 → 着色 → MP4（24fps） |
| `get_env_trial_frames` | 函数 | notebook/notebook_utils.py | 扫描帧目录，校验连续性，返回有效 trial 区间 |
| `_shade_segmentation` | Warp kernel | notebook/notebook_utils.py | 法线×光照对着色图做明暗着色（GPU） |
| `submit_job` 等 3 路由 | Flask | notebook/app.py | `/canny/submit|status|result` REST API |
| `process_video_job` | 函数 | notebook/app.py | 后台线程 subprocess 调用 cosmos_transfer1 的 transfer.py |

**调用链**：`generate_dataset.ipynb` → `notebook_widgets`(UI) / `notebook_utils`(帧处理) / `cosmos_request.process_video` → HTTP → `app.py`(Cosmos 节点, :5000) → subprocess `cosmos_transfer1/diffusion/inference/transfer.py`

## 约定

- **帧文件名模式**（强约定，多处正则依赖）：`{camera}_{modality}_trial_{n}_tile_{env}_step_{idx}.png`，modality ∈ {normals, semantic_segmentation}
- **输出目录约定**：仿真输出 `_isaaclab_out/`，Cosmos 输出 `_cosmos_out/`（定义于 notebook_utils.py 顶部常量）
- **任务 ID 固定**：`Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0`（notebook_widgets.create_task_input 硬编码）
- **双节点架构**：Isaac Lab 仿真与 Cosmos 推理必须分节点运行（硬件要求不同）

## 反模式（本项目禁止）

- ❌ 不要在容器内路径写死本地路径——所有 notebook/ 文件经 docker-compose **逐个**挂载到 `/workspace/isaaclab/`
- ❌ 不要重命名 notebook/ 下已有文件——会破坏挂载映射与 ipynb 导入
- ❌ 不要假设 launch.sh 可读——见注意事项
- ❌ 不要删除 `verify=False` 后不加说明地改请求逻辑（自签证书环境）

## 注意事项

1. **launch.sh 被 DLP 加密**（UniDocSafe/Leagsoft 透明加密）：本仓库环境中该文件内容不可读。修改前先确认在装有对应解密驱动的机器上操作。
2. **新增 notebook/ 文件时必须同步修改 docker-compose.yml** 增加 volume 条目，否则容器内看不到。
3. Cosmos 服务端 `app.py` 必须放在 cosmos_transfer1 仓库根目录运行（`CONTROL2WORLD_PATH` 为相对路径），端口 5000。
4. `cosmos_request.py` 全部请求 `verify=False`（自签证书）；轮询默认间隔 10s、上限 3600s。
5. 无测试、无 lint/format 配置、无 CI——改动靠 Notebook 实跑验证。

## 命令

```bash
# 启动（需先 xhost +local: 开启 X11 转发；Linux + NVIDIA GPU 环境）
docker compose -f docker-compose.yml up -d
# 访问 JupyterLab
# http://localhost:8888/lab/tree/generate_dataset.ipynb
# 停止
docker compose -f docker-compose.yml down
```

## 相关文档

- [docs/AGENTS-notebook.md](AGENTS-notebook.md) — notebook/ 工作流模块详解
- [docs/AGENTS-deployment.md](AGENTS-deployment.md) — 部署、硬件要求与环境配置
