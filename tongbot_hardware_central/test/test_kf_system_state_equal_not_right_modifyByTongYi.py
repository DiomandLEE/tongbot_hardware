import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

'''
从L97到L142，分别采用了不同的滤波器，对一段时间内的加速度，进行后处理的平滑
并且将从加速度推导的速度和位置，进行了更合理的计算
但是，突然想到我们使用卡尔曼滤波，就是想实时获取，所以对一段时间的后处理，不太现实
'''

# 定义卡尔曼滤波器类
class KalmanFilter:
    def __init__(self, dt, process_noise, measurement_noise):
        self.dt = dt  # 时间步长

        # 状态向量 [位置, 速度, 加速度]
        self.x = np.zeros((3, 1))  # 初始状态为零

        # 状态转移矩阵 F (匀加速模型)
        self.F = np.array([[1, dt, 0.5 * dt**2],
                           [0, 1, dt],
                           [0, 0, 1]])

        # 过程噪声协方差矩阵 Q
        self.Q = process_noise * np.array([[dt**4 / 4, dt**3 / 2, dt**2 / 2],
                                           [dt**3 / 2, dt**2, dt],
                                           [dt**2 / 2, dt, 1]])

        # 测量矩阵 H (只测量位置)
        self.H = np.array([[1, 0, 0]])

        # 测量噪声协方差矩阵 R (设置为一个常数)
        self.R = measurement_noise * np.array([[10]])

        # 初始误差协方差矩阵 P
        # self.P = np.eye(3)
        self.P = np.diag([1, 1, 10])  # 增大加速度误差的初始协方差


    def predict(self):
        # 状态预测
        self.x = np.dot(self.F, self.x)

        # 误差协方差预测
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        # 计算卡尔曼增益
        y = z - np.dot(self.H, self.x)  # 测量残差
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # 误差协方差
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # 卡尔曼增益

        # 更新状态
        self.x = self.x + np.dot(K, y)

        # 更新误差协方差
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_state(self):
        return self.x.copy()  # 返回状态的副本，避免外部修改影响内部状态

# 模拟非匀加速的真实系统
def true_system(t):
    dt = t[1] - t[0] if len(t) > 1 else 0.01  # 使用时间序列的步长
    acceleration = np.sin(0.5 * t)  # 非匀加速运动
    velocity = integrate.cumtrapz(acceleration, dx=dt, initial=0)
    position = integrate.cumtrapz(velocity, dx=dt, initial=0)
    return position, velocity, acceleration

# 测量噪声和过程噪声的设置
dt = 0.01  # 时间步长
process_noise = 1e-2  # 合理的过程噪声
measurement_noise = 1e-6 # 合理的测量噪声

# 初始化卡尔曼滤波器
kf = KalmanFilter(dt, process_noise, measurement_noise)

# 生成数据
time_steps = np.arange(0, 10, dt)
true_positions, true_velocities, true_accelerations = true_system(time_steps)

measured_positions = []
estimated_positions = []
estimated_velocities = []
estimated_accelerations = []

for i, t in enumerate(time_steps):
    # 假设我们只能测量位置，并加入一些噪声
    measured_pos = true_positions[i] + np.random.normal(0, np.sqrt(measurement_noise))
    measured_positions.append(measured_pos)

    # 卡尔曼滤波器的预测和更新步骤
    kf.predict()
    kf.update(np.array([[measured_pos]]))

    # 获取滤波后的估计状态（位置、速度、加速度）
    estimated_state = kf.get_state()
    estimated_positions.append(estimated_state[0, 0])
    estimated_velocities.append(estimated_state[1, 0])
    estimated_accelerations.append(estimated_state[2, 0])

# 低通滤波器，平滑加速度
alpha = 0.1 # 平滑因子，范围在[0, 1]之间
estimated_accelerations_smoothed = [estimated_accelerations[0]]
for i in range(1, len(estimated_accelerations)):
    smoothed_value = alpha * estimated_accelerations[i] + (1 - alpha) * estimated_accelerations_smoothed[-1]
    estimated_accelerations_smoothed.append(smoothed_value)

from scipy.signal import savgol_filter

# 应用 Savitzky-Golay 滤波器
window_size = 21  # 必须为奇数，窗口越大，平滑效果越强
poly_order = 5   # 多项式阶数，控制拟合的平滑程度
# smoothed_acceleration = savgol_filter(estimated_accelerations, window_size, poly_order)

from scipy.signal import butter, filtfilt

# 设计低通 Butterworth 滤波器
fs = 200  # 采样频率（假设 100 Hz）
cutoff = 2  # 截止频率 (Hz)
order = 4  # 滤波器阶数
b, a = butter(order, cutoff / (fs / 2), btype='low')

# 双向滤波器应用
# smoothed_acceleration = filtfilt(b, a, estimated_accelerations)

from scipy.signal import medfilt

# 应用中值滤波器
window_size = 11  # 滑动窗口大小，必须为奇数
# smoothed_acceleration = medfilt(estimated_accelerations, window_size)

# 均值滤波
def moving_average_filter(data, window_size):
    """
    实现均值滤波器（Moving Average Filter）
    :param data: 输入的信号数据
    :param window_size: 滤波窗口大小
    :return: 滤波后的信号
    """
    kernel = np.ones(window_size) / window_size
    filtered_data = np.convolve(data, kernel, mode='same')
    return filtered_data

# 设置窗口大小
window_size = 50  # 滤波窗口大小，可以根据需求调整
smoothed_acceleration = moving_average_filter(estimated_accelerations, window_size)



# 可视化结果
plt.figure(figsize=(10, 8))

# 位置对比
plt.subplot(3, 1, 1)
plt.plot(time_steps, true_positions, label="True Position", color="g")
plt.plot(time_steps, measured_positions, label="Measured Position", color="r", linestyle="--")
plt.plot(time_steps, estimated_positions, label="Estimated Position", color="b")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.legend()

# 速度对比
plt.subplot(3, 1, 2)
plt.plot(time_steps, true_velocities, label="True Velocity", color="g")
plt.plot(time_steps, estimated_velocities, label="Estimated Velocity", color="b")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend()

# 加速度对比
plt.subplot(3, 1, 3)
plt.plot(time_steps, true_accelerations, label="True Acceleration", color="g")
# plt.plot(time_steps, estimated_accelerations_smoothed, label="Estimated Smooth Acceleration", color="r")
# plt.plot(time_steps, estimated_accelerations, label="Estimated Acceleration", color="b")
plt.plot(time_steps, smoothed_acceleration, 'r-', label='Savitzky-Golay Smoothed Acceleration')

plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s^2]")
plt.legend()

plt.tight_layout()
plt.show()
