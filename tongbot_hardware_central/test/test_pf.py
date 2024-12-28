import numpy as np
import matplotlib.pyplot as plt
'''
测试了粒子滤波器，但是效果不是很好，不知道什么原因，从结果上看，可能是cahtgpt的公式错了
'''

# 时间设置
dt = 0.01  # 时间步长
T = 10      # 总时长
t_values = np.arange(0, T, dt)

# 真值：使用正弦函数模拟
true_positions = np.sin(t_values)  # 位置：sin波形
true_velocities = np.cos(t_values)  # 速度：cos波形
true_accelerations = -np.sin(t_values)  # 加速度：-sin波形

# 粒子滤波参数
N = 1000  # 粒子数目
particles = np.zeros((N, 3))  # 每个粒子包含位置、速度和加速度
weights = np.ones(N) / N  # 初始化每个粒子的权重

# 初始状态
particles[:, 0] = true_positions[0] + np.random.normal(0, 0.1, N)  # 初始位置加噪声
particles[:, 1] = true_velocities[0] + np.random.normal(0, 0.1, N)  # 初始速度加噪声
particles[:, 2] = true_accelerations[0] + np.random.normal(0, 0.1, N)  # 初始加速度加噪声

# 测量数据：位置带有噪声
measurements = true_positions + np.random.normal(0, 0.1, len(t_values))

# 粒子滤波的过程：迭代估计位置、速度和加速度
estimated_positions = np.zeros_like(t_values)
estimated_velocities = np.zeros_like(t_values)
estimated_accelerations = np.zeros_like(t_values)

for t in range(1, len(t_values)):
    # 预测阶段
    particles[:, 0] += particles[:, 1] * dt + 0.5 * particles[:, 2] * dt**2  # 更新位置
    particles[:, 1] += particles[:, 2] * dt  # 更新速度
    # 这里假设加速度保持不变（可以根据实际情况进行动态调整）

    # 更新权重（计算每个粒子与测量值的误差）
    diff = particles[:, 0] - measurements[t]
    weights = np.exp(-0.5 * (diff / 0.1)**2)  # 使用高斯误差模型

    # 防止权重全为零：加一个小常数
    weights += 1e-10
    weights /= np.sum(weights)  # 归一化权重，确保权重和为1

    # 重采样阶段：根据权重重采样粒子
    indices = np.random.choice(np.arange(N), size=N, p=weights)
    particles = particles[indices]

    # 计算当前估计的值：使用加权平均
    estimated_positions[t] = np.mean(particles[:, 0])
    estimated_velocities[t] = np.mean(particles[:, 1])
    estimated_accelerations[t] = np.mean(particles[:, 2])

# 绘图
plt.figure(figsize=(10, 8))

# 绘制位置
plt.subplot(3, 1, 1)
plt.plot(t_values, true_positions, label='True Position')
plt.plot(t_values, estimated_positions, label='Estimated Position')
plt.scatter(t_values, measurements, color='g', s=10, label="Measurements", alpha=0.5)
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.title('Position Estimation')

# 绘制速度
plt.subplot(3, 1, 2)
plt.plot(t_values, true_velocities, label='True Velocity')
plt.plot(t_values, estimated_velocities, label='Estimated Velocity')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Velocity Estimation')

# 绘制加速度
plt.subplot(3, 1, 3)
plt.plot(t_values, true_accelerations, label='True Acceleration')
plt.plot(t_values, estimated_accelerations, label='Estimated Acceleration')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s²]')
plt.title('Acceleration Estimation')

plt.tight_layout()
plt.show()
