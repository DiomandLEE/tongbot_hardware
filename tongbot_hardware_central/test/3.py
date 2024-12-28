import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.signal import butter, filtfilt

'''
当时的想法应该是开始怀疑有没有必要做卡尔曼滤波。
直接使用差值/dt的结果是不是就够用，测试对比发现，还是得用卡尔曼滤波
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

# 加权叠加多个三角函数来生成加速度
def true_system(t):
    dt = t[1] - t[0] if len(t) > 1 else 0.01  # 使用时间序列的步长

    # 三角函数参数
    freq1, amp1, phase1 = 0.5, 1.0, 0
    freq2, amp2, phase2 = 1.0, 0.5, np.pi / 4
    freq3, amp3, phase3 = 2.0, 0.2, np.pi / 2

    # 生成加速度（多个三角函数的叠加）
    acceleration = 0.1 * (
        amp1 * np.sin(freq1 * t + phase1) +
        amp2 * np.cos(freq2 * t + phase2) +
        amp3 * np.sin(freq3 * t + phase3)
    )

    # 计算速度和位置（累积积分）
    velocity = integrate.cumtrapz(acceleration, dx=dt, initial=0)
    position = integrate.cumtrapz(velocity, dx=dt, initial=0)

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
process_noise = 1e-3  # 合理的过程噪声
measurement_noise = 1e-2 # 合理的测量噪声

# 初始化卡尔曼滤波器
kf = KalmanFilter(dt, process_noise, measurement_noise)

# 生成数据
time_steps = np.arange(0, 20, dt)
true_positions, true_velocities, true_accelerations = true_system(time_steps)

measured_positions = []
estimated_positions = []
estimated_velocities = []
estimated_accelerations = []
acceleration_history = []
smoothed_accelerations = []

# 开始卡尔曼滤波和加速度估计
for i, t in enumerate(time_steps):
    # 假设我们只能测量位置，并加入一些噪声
    measured_pos = true_positions[i] + np.random.normal(0, np.sqrt(1e-3))
    measured_positions.append(measured_pos)

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

# 计算基于dt的速度和加速度
# approx_velocities = np.diff(true_positions) / dt  # 基于位置差计算速度
approx_velocities = np.diff(measured_positions) / dt  # 基于测量位置差计算速度
approx_velocities = np.concatenate(([0], approx_velocities))  # 补充第一个速度

approx_accelerations = np.diff(approx_velocities) / dt  # 基于速度差计算加速度
approx_accelerations = np.concatenate(([0], approx_accelerations))  # 补充第一个加速度

# 计算误差
position_error = np.array(estimated_positions) - true_positions
velocity_error = np.array(estimated_velocities) - true_velocities
acceleration_error = np.array(smoothed_accelerations) - true_accelerations

# # 计算基于dt的速度和加速度
# approx_velocities = np.diff(measured_positions) / dt  # 基于测量位置差计算速度
# approx_velocities = np.concatenate(([0], approx_velocities))  # 补充第一个速度

# approx_accelerations = np.diff(approx_velocities) / dt  # 基于速度差计算加速度
# approx_accelerations = np.concatenate(([0], approx_accelerations))  # 补充第一个加速度

# 绘制从第100个时间步开始的图像
plt.figure(figsize=(10, 12))

# 位置对比
plt.subplot(4, 1, 1)
plt.plot(time_steps[100:], true_positions[100:], label="True Position", color="g")
plt.plot(time_steps[100:], measured_positions[100:], label="Measured Position", color="r", linestyle="--")
plt.plot(time_steps[100:], estimated_positions[100:], label="Estimated Position", color="b")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.legend()

# 速度对比
plt.subplot(4, 1, 2)
plt.plot(time_steps[100:], true_velocities[100:], label="True Velocity", color="g")
plt.plot(time_steps[100:], approx_velocities[100:], label="Calculated Velocity (approx)", color="orange")
plt.plot(time_steps[100:], estimated_velocities[100:], label="Estimated Velocity", color="b")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend()

# 加速度对比
plt.subplot(4, 1, 3)
plt.plot(time_steps[100:], true_accelerations[100:], label="True Acceleration", color="g")
plt.plot(time_steps[100:], approx_accelerations[100:], label="Calculated Acceleration (approx)", color="orange")
plt.plot(time_steps[100:], smoothed_accelerations[100:], 'r-', label='Smoothed Acceleration (Bidirectional Filter)')
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s^2]")
plt.legend()

# 误差对比
plt.subplot(4, 1, 4)
plt.plot(time_steps[100:], position_error[100:], label="Position Error", color="r")
plt.plot(time_steps[100:], velocity_error[100:], label="Velocity Error", color="b")
plt.plot(time_steps[100:], acceleration_error[100:], label="Acceleration Error", color="orange")
plt.xlabel("Time [s]")
plt.ylabel("Error [m, m/s, m/s^2]")
plt.legend()

plt.tight_layout()
plt.show()

