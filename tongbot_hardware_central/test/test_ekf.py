import numpy as np
import matplotlib.pyplot as plt

'''
使用gpt生成的扩展卡尔曼滤波，不知道是不是gpt生成的不对，结果误差非常明显
'''

# 时间步长
dt = 0.001  # 0.1s
T = 20  # 模拟20秒

# 系统初始状态
x_init = 0  # 初始位置
y_init = 0  # 初始位置
vx_init = 1  # 初始速度
vy_init = 1  # 初始速度
ax_init = 0  # 初始加速度
ay_init = 0  # 初始加速度

# 初始化状态向量 (x, y, vx, vy, ax, ay)
x = np.array([x_init, y_init, vx_init, vy_init, ax_init, ay_init])

# 初始协方差矩阵
P = np.eye(6)

# 过程噪声协方差矩阵 Q 和测量噪声协方差矩阵 R
Q = np.eye(6) * 0.0001  # 过程噪声
R = np.eye(2) * 1e-80  # 测量噪声

# 测量矩阵 H（我们只有位置数据）
H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0]])

# 真实运动的加速度是正弦波
A = 0.1  # x方向的加速度振幅
B = 0.1  # y方向的加速度振幅
omega = 0.2  # 正弦波频率

# 运动模型的预测方程（非线性）
def f(x, t, dt):
    x_new = np.zeros_like(x)
    x_new[0] = x[0] + x[2]*dt + 0.5*x[4]*dt**2
    x_new[1] = x[1] + x[3]*dt + 0.5*x[5]*dt**2
    x_new[2] = x[2] + x[4]*dt
    x_new[3] = x[3] + x[5]*dt
    x_new[4] = A * np.sin(omega * t)  # x方向加速度
    x_new[5] = B * np.sin(omega * t)  # y方向加速度
    return x_new

# 计算雅可比矩阵（对状态方程的导数）
def jacobian_f(x, t, dt):
    F = np.eye(6)
    F[0, 2] = dt
    F[0, 4] = 0.5 * dt**2
    F[1, 3] = dt
    F[1, 5] = 0.5 * dt**2
    F[2, 4] = dt
    F[3, 5] = dt
    return F

# 卡尔曼滤波的预测和更新步骤
def kalman_filter(x, P, Q, R, H, z, t, dt):
    # 预测步骤
    x_pred = f(x, t, dt)
    F = jacobian_f(x, t, dt)
    P_pred = F @ P @ F.T + Q

    # 更新步骤
    y = z - H @ x_pred  # 测量残差
    S = H @ P_pred @ H.T + R  # 计算S
    K = P_pred @ H.T @ np.linalg.inv(S)  # 卡尔曼增益
    x_new = x_pred + K @ y  # 更新状态估计
    P_new = (np.eye(6) - K @ H) @ P_pred  # 更新协方差矩阵

    return x_new, P_new

# 生成真实运动（加速度正弦波）
true_positions = []
true_velocities = []
true_accelerations = []
measurements = []
for t in range(int(T / dt)):
    true_x = A * np.sin(omega * t * dt)
    true_y = B * np.sin(omega * t * dt)
    true_vx = A * omega * np.cos(omega * t * dt)
    true_vy = B * omega * np.cos(omega * t * dt)
    true_ax = A * omega**2 * np.sin(omega * t * dt)
    true_ay = B * omega**2 * np.sin(omega * t * dt)

    true_positions.append([true_x, true_y])
    true_velocities.append([true_vx, true_vy])
    true_accelerations.append([true_ax, true_ay])

    # 添加测量噪声
    z = np.array([true_x + np.random.randn() * 0.01, true_y + np.random.randn() * 0.01])
    measurements.append(z)

# 卡尔曼滤波估计
estimated_positions = []
estimated_velocities = []
estimated_accelerations = []

for t, z in enumerate(measurements):
    x, P = kalman_filter(x, P, Q, R, H, z, t * dt, dt)
    estimated_positions.append(x[:2])  # 只保存位置
    estimated_velocities.append(x[2:4])  # 保存速度
    estimated_accelerations.append(x[4:6])  # 保存加速度


# 将测量数据转换为 NumPy 数组
measurements = np.array(measurements)

# 可视化结果
estimated_positions = np.array(estimated_positions)
true_positions = np.array(true_positions)
estimated_velocities = np.array(estimated_velocities)
true_velocities = np.array(true_velocities)
estimated_accelerations = np.array(estimated_accelerations)
true_accelerations = np.array(true_accelerations)

# 绘制位置对比
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], label="Estimated Position")
plt.plot(true_positions[:, 0], true_positions[:, 1], 'r--', label="True Position")
plt.scatter(measurements[:, 0], measurements[:, 1], color='g', s=10, label="Measurements", alpha=0.5)
plt.legend()
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Position Estimation with EKF (True, Estimated, and Measurements)')

# 绘制速度对比
plt.subplot(3, 1, 2)
plt.plot(estimated_velocities[:, 0], label="Estimated Velocity X")
plt.plot(true_velocities[:, 0], 'r--', label="True Velocity X")
plt.plot(estimated_velocities[:, 1], label="Estimated Velocity Y")
plt.plot(true_velocities[:, 1], 'r--', label="True Velocity Y")
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Velocity Estimation with EKF (True and Estimated)')

# 绘制加速度对比
plt.subplot(3, 1, 3)
plt.plot(estimated_accelerations[:, 0], label="Estimated Acceleration X")
plt.plot(true_accelerations[:, 0], 'r--', label="True Acceleration X")
plt.plot(estimated_accelerations[:, 1], label="Estimated Acceleration Y")
plt.plot(true_accelerations[:, 1], 'r--', label="True Acceleration Y")
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s²]')
plt.title('Acceleration Estimation with EKF (True and Estimated)')

plt.tight_layout()
plt.show()
