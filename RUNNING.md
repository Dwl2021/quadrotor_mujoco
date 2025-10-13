# Quadrotor MuJoCo 运行指南

本文档介绍如何搭建运行环境并启动四旋翼 SE(3) 控制仿真。

## 1. 项目概览

- `main.py`：MuJoCo 仿真入口，加载场景并调用 `SE3Controller` 计算控制量。
- `se3_controller.py`：SE(3) 控制器实现，负责姿态与位置控制。
- `motor_mixer.py`：电机动力分配，将总推力和机体系矩转换为四个电机转速。
- `crazyfile/scene.xml`：Crazyflie 2 机模和地面场景的 MJCF 描述，需要 MuJoCo 2.2.2+。

## 2. 环境准备

1. **系统依赖**
   - 64 位 Linux / macOS / Windows。
   - Python 3.9 及以上（推荐 3.10+）。
   - OpenGL 图形环境（本地桌面）或支持 EGL 的 GPU/驱动（远程服务器）。

2. **可选系统包（以 Debian/Ubuntu 为例）**
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip libgl1 libglew-dev libosmesa6
   ```

3. **创建虚拟环境（推荐）**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```

4. **安装 Python 依赖**
   ```bash
   pip install mujoco numpy
   ```
   首次导入 `mujoco` 时会自动下载运行时二进制文件，需确保网络或事先手动安装。

5. **图形后端配置**
   - 本地桌面：直接运行即可。
   - 远程无显示服务器：在启动前设置 `export MUJOCO_GL=egl`，并确保驱动支持 EGL。

## 3. 运行仿真

1. 激活虚拟环境（若已创建）：
   ```bash
   source .venv/bin/activate
   ```
2. 运行仿真脚本：
   ```bash
   python main.py
   ```
3. MuJoCo Viewer 打开后：
   - `Space` 暂停/继续仿真；
   - `V` 切换渲染模式；
   - 右键拖动调节相机，鼠标滚轮缩放。

出现 `ImportError: mujoco` 时请确认依赖安装；若提示 OpenGL/EGL 错误，请检查步骤 5 中的图形后端配置。

## 4. 修改飞行任务

- 默认目标位置为 `main.py` 中 `goal_pos = [0, 0, 0.3]`，若需改为圆形轨迹，可取消注释：
  ```python
  goal_pos, goal_heading = simple_trajectory(d.time)
  ```
- `ctrl.kx`, `ctrl.kv`, `ctrl.kR`, `ctrl.kw` 控制器增益定义在 `main.py` 49-52 行；根据需求调整后重新运行脚本。
- 混控器参数位于 `motor_mixer.py`，如需适配其他机型，可修改推力/扭矩系数与臂长。

## 5. 常见问题

- **下载 MuJoCo 运行库失败**：提前手动下载 [MuJoCo](https://mujoco.org/) 对应版本，并按照官方说明配置环境变量。
- **Viewer 无法显示**：确保本地具备 OpenGL；在远程环境使用 `MUJOCO_GL=egl`；如需离屏渲染，可使用 `mujoco.viewer.launch_passive` 并自定义渲染。
- **仿真不稳定**：根据需要调低 `dt`、调节控制增益或在 `simple_trajectory` 中降低半径/速度参数。

至此即可在本地启动并体验四旋翼 SE(3) 控制仿真。若要进一步集成到其他框架，建议封装 `SE3Controller` 类并复用现有状态与命令接口。
