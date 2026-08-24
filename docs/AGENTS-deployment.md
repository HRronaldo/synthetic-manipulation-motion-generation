# AGENTS.md — 部署与环境

> 父文档：[docs/AGENTS.md](AGENTS.md)。本文件只写部署/硬件/环境配置，不重复父级。

## 概述

单容器 Docker Compose 部署（Isaac Lab 仿真节点）+ 独立 H100 节点运行 Cosmos 推理服务。双节点缺一不可。

## 硬件与系统要求

| 组件 | 要求 |
|------|------|
| 仿真节点 OS | Ubuntu 22.04 |
| 仿真节点 GPU | RTX A6000 48GB（驱动 ≥535.129.03） |
| Cosmos 节点 GPU | H100 80GB+（AWS P5 / GCP A3 / Azure ND H100 v5） |
| Docker | + NVIDIA Container Toolkit ≥1.17.0 |

## docker-compose.yml 要点

- 镜像：`nvcr.io/nvidia/gr00t-smmg-bp:1.0`，`pull_policy: always`，内含 Isaac Lab 2.0.2 + Isaac Sim 4.5.0
- `privileged: true` + `network_mode: host` + `runtime: nvidia`（GPU 全量直通）
- X11 显示：挂载 `/tmp/.X11-unix`、`$HOME/.Xauthority`，需先 `xhost +local:`
- 入口点：`/workspace/isaaclab/launch.sh`（由本地 `./launch.sh` 挂载）
- **逐文件挂载**：notebook/ 下 6 个文件各自一条 volume → `/workspace/isaaclab/`
- 数据集：`samples/annotated_dataset.hdf5` → 容器内 `/workspace/isaaclab/datasets/annotated_dataset.hdf5`
- 缓存持久化：`$HOME/docker/isaac-lab/cache/kit` → `/isaac-sim/kit/cache`

## 命令

```bash
xhost +local:                                        # 1. 开 X11
docker compose -f docker-compose.yml up -d           # 2. 启动
# 浏览器打开 http://localhost:8888/lab/tree/generate_dataset.ipynb
docker compose -f docker-compose.yml down            # 3. 停止
```

Cosmos 节点侧（H100）：在 cosmos_transfer1 仓库根目录运行 `python app.py`（Flask :5000），Notebook 中填该节点 IP。

## 反模式

- ❌ 不要用 `ports:` 映射替代 host 网络——Livestream/JupyterLab 端口注释仅作参考，host 模式下直接访问
- ❌ 不要删 `ACCEPT_EULA: Y`——镜像拉取/启动需要
- ❌ 不要把两个角色部署到同一台机器——README 明确要求分节点

## 注意事项

- `launch.sh` 被 DLP 透明加密（UniDocSafe/Leagsoft），非授权环境读取为乱码；它是容器入口点，损坏 = 容器起不来
- 运行 docker compose 即视为接受 README 所列全部许可（Isaac Sim/Lab/mimic/Cosmos）
- `restart: unless-stopped`——容器崩溃会自动重启，排查问题前先看 `docker logs`
