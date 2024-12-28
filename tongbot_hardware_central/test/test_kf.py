# import numpy as np
# import matplotlib.pyplot as plt

'''
相当简单的卡尔曼测试，假定的true system，是非常离散的，pass掉
'''

# # 参数设置
# dt = 1.0  # 时间间隔
# F = np.array([[1, dt, 0.5*dt**2], [0, 1, dt], [0, 0, 1]])
# H = np.array([[1, 0, 0]])
# Q = np.eye(3) * 0.001  # 过程噪声协方差
# R = np.array([[1.0]])  # 测量噪声协方差，注意这里传递的是一个列表 R = np.array() * 1.0  # 测量噪声协方差

# # 初始状态和误差协方差
# x_hat = np.array([0, 0, 0]).reshape(-1, 1)
# P = np.eye(3) * 10  # 较大的初始误差协方差

# # 测量数据
# measurements = np.array([0, 1.9, 4.5, 8.6, 13.7, 19.6, 26.5, 34.4, 43.2, 52.9])

# # 存储估计结果
# positions_estimated = []
# velocities_estimated = []
# accelerations_estimated = []

# for z in measurements:
#     # 预测步骤
#     x_hat = F @ x_hat
#     P = F @ P @ F.T + Q

#     # 更新步骤
#     K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
#     x_hat = x_hat + K @ (np.array([[z]]) - H @ x_hat)
#     P = (np.eye(3) - K @ H) @ P

#     # 存储估计结果
#     positions_estimated.append(x_hat[0, 0])
#     velocities_estimated.append(x_hat[1, 0])
#     accelerations_estimated.append(x_hat[2, 0])

# # 可视化结果
# time_steps = np.arange(len(measurements))

# plt.figure(figsize=(12, 9))

# # 绘制位置
# plt.subplot(3, 1, 1)
# plt.plot(time_steps, measurements, 'rx', label='Measured Positions')
# plt.plot(time_steps, positions_estimated, 'b-', label='Estimated Positions')
# plt.title('Position Estimation')
# plt.xlabel('Time Step')
# plt.ylabel('Position (m)')
# plt.legend()

# # 绘制速度
# plt.subplot(3, 1, 2)
# plt.plot(time_steps, velocities_estimated, 'g-', label='Estimated Velocities')
# plt.title('Velocity Estimation')
# plt.xlabel('Time Step')
# plt.ylabel('Velocity (m/s)')
# plt.legend()

# # 绘制加速度
# plt.subplot(3, 1, 3)
# plt.plot(time_steps, accelerations_estimated, 'y-', label='Estimated Accelerations')
# plt.title('Acceleration Estimation')
# plt.xlabel('Time Step')
# plt.ylabel('Acceleration (m/s^2)')
# plt.legend()

# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 参数设置
dt = 1.0  # 时间间隔
F = np.array([[1, dt, 0.5*dt**2], [0, 1, dt], [0, 0, 1]])
H = np.array([[1, 0, 0]])
Q = np.eye(3) * 0.001  # 减小过程噪声
R = np.array([[1.0]])  # 测量噪声协方差

# 初始状态和误差协方差
x_hat = np.array([0, 0, 9.8]).reshape(-1, 1)  # 初始加速度设为9.8 m/s^2
P = np.eye(3) * 1000  # 较大的初始误差协方差

# 测量数据（模拟更多时间步）
np.random.seed(0)  # 设置随机种子以获得可重复的结果
true_acceleration = 9.8  # 真实加速度
time_steps = np.arange(20)
positions_true = 0.5 * true_acceleration * time_steps**2
measurements = positions_true + np.random.normal(0, 1, size=time_steps.shape)

# 存储估计结果
positions_estimated = []
velocities_estimated = []
accelerations_estimated = []

for z in measurements:
    # 预测步骤
    x_hat = F @ x_hat
    P = F @ P @ F.T + Q

    # 更新步骤
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x_hat = x_hat + K @ (np.array([[z]]) - H @ x_hat)
    P = (np.eye(3) - K @ H) @ P

    # 存储估计结果
    positions_estimated.append(x_hat[0, 0])
    velocities_estimated.append(x_hat[1, 0])
    accelerations_estimated.append(x_hat[2, 0])

# 可视化结果
plt.figure(figsize=(12, 9))

# 绘制位置
plt.subplot(3, 1, 1)
plt.plot(time_steps, measurements, 'rx', label='Measured Positions')
plt.plot(time_steps, positions_estimated, 'b-', label='Estimated Positions')
plt.title('Position Estimation')
plt.xlabel('Time Step')
plt.ylabel('Position (m)')
plt.legend()

# 绘制速度
plt.subplot(3, 1, 2)
plt.plot(time_steps, velocities_estimated, 'g-', label='Estimated Velocities')
plt.title('Velocity Estimation')
plt.xlabel('Time Step')
plt.ylabel('Velocity (m/s)')
plt.legend()

# 绘制加速度
plt.subplot(3, 1, 3)
plt.plot(time_steps, accelerations_estimated, 'y-', label='Estimated Accelerations')
plt.axhline(y=true_acceleration, color='r', linestyle='--', label='True Acceleration')
plt.title('Acceleration Estimation')
plt.xlabel('Time Step')
plt.ylabel('Acceleration (m/s^2)')
plt.legend()

plt.tight_layout()
plt.show()