import numpy as np
import matplotlib.pyplot as plt
from collections import deque

'''
这个是用滑动窗口来解决，把一段时间内的数据都用来更新状态，但是卡尔曼的假设就是当前时刻，只与上一时刻有关，所以也不行
'''

class KalmanFilter:
    def __init__(self, dt, process_noise, measurement_noise, window_size):
        self.dt = dt
        self.window_size = window_size  # 设置滑动窗口的大小
        self.x = np.zeros((3, 1))  # 状态向量 [位置, 速度, 加速度]
        self.F = np.array([[1, dt, 0.5 * dt**2], [0, 1, dt], [0, 0, 1]])  # 状态转移矩阵
        self.Q = process_noise * np.array([[dt**4 / 4, dt**3 / 2, dt**2 / 2], [dt**3 / 2, dt**2, dt], [dt**2 / 2, dt, 1]])  # 过程噪声协方差矩阵
        self.H = np.array([[1, 0, 0]])  # 测量矩阵
        self.R = measurement_noise * np.array([[1]])  # 测量噪声协方差矩阵
        self.P = np.eye(3)  # 误差协方差矩阵
        self.window = deque(maxlen=window_size)  # 使用队列实现滑动窗口

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)  # 测量预测残差
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # 测量残差协方差
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # 卡尔曼增益
        self.x = self.x + np.dot(K, y)  # 更新状态
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)  # 更新误差协方差矩阵

    def get_state(self):
        return self.x

    def add_to_window(self, z):
        # 向滑动窗口添加新数据
        self.window.append(z)
        # 根据窗口中的数据更新状态和卡尔曼滤波
        if len(self.window) == self.window_size:
            self.predict()
            # 用窗口中的数据更新
            for measurement in self.window:
                self.update(measurement)

# 模拟非匀加速的真实系统（加速度是正弦波）
def true_system(t):
    acceleration = 0.1 * np.sin(0.5 * t)  # 加速度是正弦变化
    velocity = 0.1 * np.cos(0.5 * t)  # 速度是加速度的积分
    position = 0.1 * np.sin(0.5 * t)  # 位置是速度的积分
    return position, velocity, acceleration

# 时间步长和噪声设定
dt = 0.1  # 时间步长
process_noise = 1e-80  # 过程噪声较小，允许更多依赖测量值
measurement_noise = 1e-40
window_size = 5  # 设置滑动窗口的大小

kf = KalmanFilter(dt, process_noise, measurement_noise, window_size)

# 采样时间和状态存储
time_steps = np.arange(0, 10, dt)
true_positions = []
measured_positions = []
estimated_positions = []
estimated_velocities = []
estimated_accelerations = []

for t in time_steps:
    true_pos, true_vel, true_acc = true_system(t)
    true_positions.append(true_pos)

    # 模拟带噪声的测量
    measured_pos = true_pos + np.random.normal(0, measurement_noise)
    measured_positions.append(measured_pos)

    # 使用滑动窗口进行卡尔曼滤波
    kf.add_to_window(np.array([[measured_pos]]))

    estimated_state = kf.get_state()
    estimated_positions.append(estimated_state[0, 0])
    estimated_velocities.append(estimated_state[1, 0])
    estimated_accelerations.append(estimated_state[2, 0])

# 可视化结果
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(time_steps, true_positions, label="True Position", color="g")
plt.plot(time_steps, measured_positions, label="Measured Position", color="r", linestyle="--")
plt.plot(time_steps, estimated_positions, label="Estimated Position", color="b")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time_steps, np.gradient(true_positions, dt), label="True Velocity", color="g")
plt.plot(time_steps, estimated_velocities, label="Estimated Velocity", color="b")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time_steps, np.gradient(np.gradient(true_positions, dt), dt), label="True Acceleration", color="g")
plt.plot(time_steps, estimated_accelerations, label="Estimated Acceleration", color="b")
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s^2]")
plt.legend()

plt.tight_layout()
plt.show()
