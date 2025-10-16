# Quadrotor MuJoCo

## 快速训练 / 评估碰撞回弹策略
- 训练：`conda run -n torch_env env MUJOCO_GL=osmesa ./train.sh --episodes 2000`
- 评估：`conda run -n torch_env env MUJOCO_GL=osmesa ./eval.sh --episodes 100`
- 可视化评估策略：`conda run -n torch_env env MUJOCO_GL=glfw ./eval.sh --episodes 5 --render --sleep 0.02`
- 可视化基线（无碰撞策略）：`conda run -n torch_env env MUJOCO_GL=glfw ./eval.sh --episodes 5 --render --baseline --sleep 0.02`

训练结果会保存到 `policies/collision_policy.npz`，评估脚本会打印平均奖励、成功率等指标。根据需要调整 `--episodes` 次数即可。
