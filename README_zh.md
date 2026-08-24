# NVIDIA Omniverse 蓝图：机器人合成操作动作生成（Synthetic Manipulation Motion Generation for Robotics）

NVIDIA Isaac GR00T 合成操作动作生成蓝图是理想的入门之选。这是一套参考工作流，用于基于少量人类示范（human demonstrations），以指数级规模生成海量的机器人操作合成运动轨迹。该蓝图构建于 [NVIDIA Omniverse™](https://developer.nvidia.com/isaac/sim) 和 [NVIDIA Cosmos™](https://www.nvidia.com/en-us/ai/cosmos/) 之上。

![image](https://github.com/user-attachments/assets/f3621fcc-91c3-4f4d-a516-c9c9c7f0d339)


# 在本地工作站上部署

## 前置条件
**本地部署要求：**
* Ubuntu 22.04 操作系统
* NVIDIA GPU（RTX A 6000，48GB 显存）
* [NVIDIA GPU 驱动](https://www.nvidia.com/en-us/drivers/unix/)（推荐版本 535.129.03）
* [Docker](https://docs.docker.com/engine/install/ubuntu/)
* [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)（最低版本 1.17.0）

**NVIDIA Cosmos 要求：**
* NVIDIA GPU（H100 或更高），80GB 显存。
  * NVIDIA H100 GPU 可在以下云平台获取：AWS 的 P5 EC2 实例、GCP 的 A3 机型虚拟机、Azure 的 ND H100 v5 系列虚拟机
* 具体详情请访问 [Cosmos Hugging Face 模型页面](https://huggingface.co/nvidia/Cosmos-Transfer1-7B)
>[!NOTE]
由于硬件需求不同，NVIDIA Cosmos 必须运行在与 Isaac Lab 仿真分离的独立节点上。

## 启动 Jupyter Notebook

步骤：

1. 克隆本仓库到本地工作站，并进入仓库目录。

       git clone https://github.com/NVIDIA-Omniverse-blueprints/synthetic-manipulation-motion-generation.git
       cd synthetic-manipulation-motion-generation

2. 为本地工作站用户启用 X11 转发。

       xhost +local:

3. 使用蓝图容器部署 Jupyter Notebook。

       docker compose -f docker-compose.yml up -d
       
4. 在浏览器中访问 http://localhost:8888/lab/tree/generate_dataset.ipynb 打开 Jupyter Notebook。

5. 按照 Jupyter Notebook 内的说明操作。

6. 运行以下命令停止 Jupyter Notebook 并结束演示。

       docker compose -f docker-compose.yml down

>[!NOTE]
蓝图容器内置了预装好的 Isaac Lab 2.0.2 和 Isaac Sim 4.5.0。

# 许可证

运行 docker compose 命令即表示您接受以下所有许可证的条款与条件：

- [Isaac Sim](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab/blob/main/LICENSE)
- [Isaac Lab mimic](https://github.com/isaac-sim/IsaacLab/blob/main/LICENSE-mimic)
- [Cosmos NVIDIA 开放模型许可协议](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
