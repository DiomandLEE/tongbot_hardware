import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.signal import butter, filtfilt

from scipy.signal import savgol_filter
import collections

'''
思路是既然使用了双向滤波后处理的话，那么就对之前的加速度估计结果进行后处理平滑，
然后将之前的加速度结果更新，看看是不是能起到后处理的效果
结果发现一般，而且会出现，加速度结果的滞后性

又测试了一下saygol_filter，发现效果很差

Savitzky-Golay 滤波器测试，与之前的卡尔曼滤波1.py对比

'''
class RealTimeFilter:
    def __init__(self, window_size=31, polyorder=2, delta=1.0):
        self.window_size = window_size
        self.polyorder = polyorder
        self.delta = delta

        # 滑动窗口，用于保存最近的 window_size 个数据点
        self.positions = collections.deque(maxlen=window_size)
        self.timestamps = collections.deque(maxlen=window_size)
        self.vel = []
        self.acc = []

    def update(self, timestamp, position):
        """每次新数据到达时更新窗口，并计算速度与加速度"""
        # 更新时间戳和位置数据
        self.timestamps.append(timestamp)
        self.positions.append(position)

        # 如果数据窗口不满，就不计算
        if len(self.positions) < self.window_size:
            self.vel.append(0)
            self.acc.append(0)
            return self.positions[-1], self.vel[-1], self.acc[-1]

        # 将时间戳差异 (dt) 用于计算速度和加速度
        dt = np.mean(np.diff(self.timestamps))  # 假设时间间隔是均匀的

        # 使用 Savitzky-Golay 滤波器计算平滑的速度和加速度
        smooth_positions = np.array(self.positions)
        smooth_velocities = savgol_filter(smooth_positions, self.window_size, self.polyorder, deriv=1, delta=dt, axis=0)
        smooth_accelerations = savgol_filter(smooth_positions, self.window_size, self.polyorder, deriv=2, delta=dt, axis=0)

        return smooth_positions[-1], smooth_velocities[-1], smooth_accelerations[-1]

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
        self.Q = process_noise * np.array([[dt**4 / 4, 1 * dt**3 / 2, 1 * dt**2 / 2],
                                           [1 * dt**3 / 2, 1 * dt**2, 1 * dt],
                                           [1 *dt**2 / 2, 1* dt, 1]])
        # self.Q = np.ones((3, 3)) * (dt**2 / 4)

        # 测量矩阵 H (只测量位置)
        self.H = np.array([[1, 0, 0]])

        # 测量噪声协方差矩阵 R (设置为一个常数)
        self.R = measurement_noise * np.array([[10]])

        # 初始误差协方差矩阵 P
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

# # 模拟非匀加速的真实系统
# def true_system(t):
#     dt = t[1] - t[0] if len(t) > 1 else 0.01  # 使用时间序列的步长
#     acceleration = np.sin(0.5 * t)  # 非匀加速运动
#     velocity = integrate.cumtrapz(acceleration, dx=dt, initial=0)
#     position = integrate.cumtrapz(velocity, dx=dt, initial=0)
#     return position, velocity, acceleration

# def true_system(t):
#     dt = t[1] - t[0] if len(t) > 1 else 0.01  # 使用时间序列的步长

#     # 添加随机噪声来模拟不规律的加速度
#     # 使用随机过程生成加速度（如正态分布的噪声）：
#     noise_amplitude = 0.8  # 调整噪声幅度
#     random_acceleration = np.sin(0.5 * t) + noise_amplitude * np.random.normal(0, 1, size=t.shape)

#     # 计算速度和位置（累积积分）
#     velocity = integrate.cumtrapz(random_acceleration, dx=dt, initial=0)
#     position = integrate.cumtrapz(velocity, dx=dt, initial=0)

#     return position, velocity, random_acceleration
# def true_system(t):
#     dt = t[1] - t[0] if len(t) > 1 else 0.01  # 使用时间序列的步长

#     # 使用一个平滑的随机噪声模型来扰动加速度
#     noise_amplitude = 0.005  # 控制噪声的强度，调低噪声幅度
#     random_walk = np.cumsum(np.random.normal(0, noise_amplitude, size=t.shape))  # 随机行走
#     random_acceleration = np.sin(0.5 * t) + random_walk  # 原有规律的加速度 + 随机行走噪声

#     # 计算速度和位置（累积积分）
#     velocity = integrate.cumtrapz(random_acceleration, dx=dt, initial=0)
#     position = integrate.cumtrapz(velocity, dx=dt, initial=0)

#     return position, velocity, random_acceleration
def true_system(t):
    dt = t[1] - t[0] if len(t) > 1 else 0.01  # 使用时间序列的步长

    # 加权叠加多个三角函数来生成加速度
    # 三角函数的频率、幅值和相位调整
    freq1, amp1, phase1 = 0.5, 1.0, 0  # 第一个三角函数
    freq2, amp2, phase2 = 1.0, 0.5, np.pi / 4  # 第二个三角函数
    freq3, amp3, phase3 = 2.0, 0.2, np.pi / 2  # 第三个三角函数

    # 生成加速度（多个三角函数的叠加）
    acceleration = 0.2 * (
        amp1 * np.sin(freq1 * t + phase1) +
        amp2 * np.cos(freq2 * t + phase2) +
        amp3 * np.sin(freq3 * t + phase3)
    )

    noise_amplitude = 0.05  # 调整噪声幅度
#     random_acceleration = np.sin(0.5 * t) + noise_amplitude * np.random.normal(0, 1, size=t.shape)


    # 计算速度和位置（累积积分）
    velocity = integrate.cumtrapz(acceleration, dx=dt, initial=0) + 0.5 * noise_amplitude * np.random.normal(0, 1, size=t.shape)
    position = integrate.cumtrapz(velocity, dx=dt, initial=0) + noise_amplitude * np.random.normal(0, 1, size=t.shape)

    return position, velocity, acceleration



# 双向滤波器
def bidirectional_filter(data, b, a):
    # 确保数据量足够进行滤波
    min_data_length = 2 * max(len(b), len(a))  # 双向滤波器所需的最小数据量
    if len(data) < min_data_length:
        print(f"Warning: Not enough data to apply bidirectional filter. Data length: {len(data)}")
        return data  # 如果数据不足，直接返回原数据
    else:
        # 对数据进行双向滤波
        try:
            return filtfilt(b, a, filtfilt(b, a, data)[::-1])[::-1]
        except ValueError as e:
            print(f"Error applying bidirectional filter: {e}")
            return data

# 设置滤波器参数
fs = 100  # 采样频率（假设 100 Hz）
cutoff = 5 # 截止频率 (Hz)
order = 2  # 滤波器阶数
b, a = butter(order, cutoff / (fs / 2), btype='low')

# 时间设置
dt = 0.01  # 时间步长
process_noise = 1e-4 # 合理的过程噪声
measurement_noise = 1e-4 # 合理的测量噪声
# 初始化卡尔曼滤波器
kf = KalmanFilter(dt, process_noise, measurement_noise)

# 生成数据
time_steps = np.arange(0, 10, dt)
time_steps1 = np.arange(0, 10, dt)
true_positions, true_velocities, true_accelerations = true_system(time_steps)

measured_positions = []
estimated_positions = []
estimated_velocities = []
estimated_accelerations = []
acceleration_history = []
smoothed_accelerations = []

savitzky_golay_filter_pos = []
savitzky_golay_filter_vel = []
savitzky_golay_filter_acc = []

# 使用示例
filter = RealTimeFilter(window_size=31, polyorder=5, delta=0.1)

# 开始卡尔曼滤波和加速度估计
for i, t in enumerate(time_steps):
    # 假设我们只能测量位置，并加入一些噪声
    measured_pos = true_positions[i] + np.random.normal(0, np.sqrt(1e-8))
    measured_positions.append(measured_pos)

    time_stamp = i * dt
    # 更新滤波器并计算速度和加速度
    positions, velocities, accelerations = filter.update(time_stamp, measured_pos)

    savitzky_golay_filter_pos.append(positions)
    savitzky_golay_filter_vel.append(velocities)
    savitzky_golay_filter_acc.append(accelerations)

    # 卡尔曼滤波器的预测和更新步骤
    kf.predict()
    kf.update(np.array([[measured_pos]]))

    # 获取滤波后的估计状态（位置、速度、加速度）
    estimated_state = kf.get_state()
    estimated_positions.append(estimated_state[0, 0])
    estimated_velocities.append(estimated_state[1, 0])
    estimated_accelerations.append(estimated_state[2, 0])

    # 存储实时加速度估计值
    acceleration_history.append(estimated_state[2, 0])

    # 当数据量足够时，才对加速度历史数据进行双向滤波
    if len(acceleration_history) >= 2 * max(len(b), len(a)):
        smoothed_acceleration = bidirectional_filter(acceleration_history, b, a)
        smoothed_accelerations.append(smoothed_acceleration[-1])  # 取最后一个元素作为实时估计值
        #acceleration_history移除最新的加速度数据
        acceleration_history.pop()
        acceleration_history.append(smoothed_acceleration[-1])
    else:
        smoothed_accelerations.append(estimated_state[2, 0])  # 初始时刻直接使用加速度估计


# 计算误差
position_error = np.array(estimated_positions) - true_positions
velocity_error = np.array(estimated_velocities) - true_velocities
acceleration_error = np.array(smoothed_accelerations) - true_accelerations

# 可视化结果
plt.figure(figsize=(10, 12))

true_velocities1 = true_velocities
true_accelerations1 = true_accelerations

time_steps = time_steps[100:]
true_positions = true_positions[100:]
true_velocities = true_velocities[100:]
true_accelerations = true_accelerations[100:]
measured_positions = measured_positions[100:]
estimated_positions = estimated_positions[100:]
estimated_velocities = estimated_velocities[100:]
smoothed_accelerations = smoothed_accelerations[100:]
# 位置对比
plt.subplot(6, 1, 1)
plt.plot(time_steps, true_positions, label="True Position", color="g")
plt.plot(time_steps, measured_positions, label="Measured Position", color="r", linestyle="--")
plt.plot(time_steps, estimated_positions, label="Estimated Position", color="b")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.legend()

# 速度对比
plt.subplot(6, 1, 2)
plt.plot(time_steps, true_velocities, label="True Velocity", color="g")
plt.plot(time_steps, estimated_velocities, label="Estimated Velocity", color="b")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend()

# 加速度对比
plt.subplot(6, 1, 3)
plt.plot(time_steps, true_accelerations, label="True Acceleration", color="g")
plt.plot(time_steps, smoothed_accelerations, 'r-', label='Smoothed Acceleration (Bidirectional Filter)')
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s^2]")
plt.legend()

# 误差对比
plt.subplot(6, 1, 4)
plt.plot(time_steps, position_error[100:], label="Position Error", color="r")
plt.plot(time_steps, velocity_error[100:], label="Velocity Error", color="b")
plt.plot(time_steps, acceleration_error[100:], label="Acceleration Error", color="orange")
plt.xlabel("Time [s]")
plt.ylabel("Error [m, m/s, m/s^2]")
plt.legend()

# 速度对比
plt.subplot(6, 1, 5)
plt.plot(time_steps1, true_velocities1, label="True Velocity", color="g")
plt.plot(time_steps1, savitzky_golay_filter_vel, label="Estimated Velocity", color="b")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend()

# 加速度对比
plt.subplot(6, 1, 6)
plt.plot(time_steps1, true_accelerations1, label="True Acceleration", color="g")
plt.plot(time_steps1, savitzky_golay_filter_acc, 'r-', label='Smoothed Acceleration (Bidirectional Filter)')
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s^2]")
plt.legend()

plt.tight_layout()
plt.show()
