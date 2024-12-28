# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import integrate

'''
这是看到之前的代码结果，都是速度估计得比较准，
所以升高状态到匀加加速度运动，看看是不是加速度会好一些，但是结果还是一般，需要靠滤波后处理
'''

# class KalmanFilter:
#     def __init__(self, dt, process_noise, measurement_noise):
#         self.dt = dt  # 时间步长

#         # 状态向量 [位置, 速度, 加速度, 加速度变化率]
#         self.x = np.zeros((4, 1))  # 初始状态为零

#         # 状态转移矩阵 F (引入 jerk 作为加速度变化率)
#         self.F = np.array([[1, dt, 0.5 * dt**2, (dt**3) / 6],
#                            [0, 1, dt, 0.5 * dt**2],
#                            [0, 0, 1, dt],
#                            [0, 0, 0, 1]])

#         # 过程噪声协方差矩阵 Q
#         self.Q = process_noise * np.array([[dt**4 / 4, dt**3 / 2, dt**2 / 2, dt**3 / 6],
#                                            [dt**3 / 2, dt**2, dt, dt**2 / 2],
#                                            [dt**2 / 2, dt, 1, dt],
#                                            [dt**3 / 6, dt**2 / 2, dt, 1]])

#         # 测量矩阵 H (只测量位置)
#         self.H = np.array([[1, 0, 0, 0]])

#         # 测量噪声协方差矩阵 R
#         self.R = measurement_noise * np.array([[1]])

#         # 初始误差协方差矩阵 P
#         self.P = np.diag([1, 1, 10, 15])

#     def predict(self):
#         # 状态预测
#         self.x = np.dot(self.F, self.x)

#         # 误差协方差预测
#         self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

#     def update(self, z):
#         # 计算卡尔曼增益
#         y = z - np.dot(self.H, self.x)  # 测量残差
#         S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # 误差协方差
#         K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # 卡尔曼增益

#         # 更新状态
#         self.x = self.x + np.dot(K, y)

#         # 更新误差协方差
#         I = np.eye(self.P.shape[0])
#         self.P = np.dot(I - np.dot(K, self.H), self.P)

#     def get_state(self):
#         return self.x.copy()  # 返回状态的副本，避免外部修改影响内部状态

# # 模拟非匀加速的真实系统
# def true_system(t):
#     # dt = t[1] - t[0] if len(t) > 1 else 0.01  # 使用时间序列的步长
#     acceleration = np.sin(0.5 * t)  # 非匀加速运动
#     jerk = np.cos(0.5 * t)  # jerk (加速度变化率)
#     velocity = integrate.cumtrapz(acceleration, dx=dt, initial=0)
#     position = integrate.cumtrapz(velocity, dx=dt, initial=0)
#     return position, velocity, acceleration, jerk

# # 测量噪声和过程噪声的设置
# dt = 0.01  # 时间步长
# process_noise = 1e-2  # 合理的过程噪声
# measurement_noise = 1e-6  # 合理的测量噪声

# # 初始化卡尔曼滤波器
# kf = KalmanFilter(dt, process_noise, measurement_noise)

# # 生成数据
# time_steps = np.arange(0, 10, dt)
# true_positions, true_velocities, true_accelerations, true_jerks = true_system(time_steps)

# measured_positions = []
# estimated_positions = []
# estimated_velocities = []
# estimated_accelerations = []
# estimated_jerks = []

# for i, t in enumerate(time_steps):
#     # 假设我们只能测量位置，并加入一些噪声
#     measured_pos = true_positions[i] + np.random.normal(0, np.sqrt(measurement_noise))
#     measured_positions.append(measured_pos)

#     # 卡尔曼滤波器的预测和更新步骤
#     kf.predict()
#     kf.update(np.array([[measured_pos]]))

#     # 获取滤波后的估计状态（位置、速度、加速度、jerk）
#     estimated_state = kf.get_state()
#     estimated_positions.append(estimated_state[0, 0])
#     estimated_velocities.append(estimated_state[1, 0])
#     estimated_accelerations.append(estimated_state[2, 0])
#     estimated_jerks.append(estimated_state[3, 0])

# # 可视化结果
# plt.figure(figsize=(10, 8))

# # 位置对比
# plt.subplot(4, 1, 1)
# plt.plot(time_steps, true_positions, label="True Position", color="g")
# plt.plot(time_steps, measured_positions, label="Measured Position", color="r", linestyle="--")
# plt.plot(time_steps, estimated_positions, label="Estimated Position", color="b")
# plt.xlabel("Time [s]")
# plt.ylabel("Position [m]")
# plt.legend()

# # 速度对比
# plt.subplot(4, 1, 2)
# plt.plot(time_steps, true_velocities, label="True Velocity", color="g")
# plt.plot(time_steps, estimated_velocities, label="Estimated Velocity", color="b")
# plt.xlabel("Time [s]")
# plt.ylabel("Velocity [m/s]")
# plt.legend()

# # 均值滤波
# def moving_average_filter(data, window_size):
#     """
#     实现均值滤波器（Moving Average Filter）
#     :param data: 输入的信号数据
#     :param window_size: 滤波窗口大小
#     :return: 滤波后的信号
#     """
#     kernel = np.ones(window_size) / window_size
#     filtered_data = np.convolve(data, kernel, mode='same')
#     return filtered_data

# # 设置窗口大小
# window_size = 50  # 滤波窗口大小，可以根据需求调整
# smoothed_acceleration = moving_average_filter(estimated_accelerations, window_size)

# # 可视化结果

# # 加速度对比
# plt.subplot(4, 1, 3)
# plt.plot(time_steps, true_accelerations, label="True Acceleration", color="g")
# plt.plot(time_steps, estimated_accelerations, label="Estimated Acceleration", color="b")
# plt.plot(time_steps, smoothed_acceleration, 'r-', label='Smoothed Acceleration')

# plt.xlabel("Time [s]")
# plt.ylabel("Acceleration [m/s^2]")
# plt.legend()

# # jerk对比
# plt.subplot(4, 1, 4)
# plt.plot(time_steps, true_jerks, label="True Jerk", color="g")
# plt.plot(time_steps, estimated_jerks, label="Estimated Jerk", color="b")
# plt.xlabel("Time [s]")
# plt.ylabel("Jerk [m/s^3]")
# plt.legend()

# plt.tight_layout()
# plt.show()

'''
这次好像是实时滤波，但是效果也不行，加速度误差还在
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

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

# 实时均值滤波的实现
class RealTimeMeanFilter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.window = []

    def update(self, new_value):
        # 将新值加入队列
        self.window.append(new_value)

        # 保证队列长度不超过指定窗口大小
        if len(self.window) > self.window_size:
            self.window.pop(0)

        # 计算并返回当前窗口的均值
        return np.mean(self.window)

# 设置均值滤波窗口大小
mean_filter = RealTimeMeanFilter(window_size=50)

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

    # 实时更新加速度的均值滤波
    smoothed_acceleration = mean_filter.update(estimated_state[2, 0])
    # 可以直接保存 `smoothed_acceleration` 如果需要在实时中使用它

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
plt.plot(time_steps, estimated_accelerations, label="Estimated Acceleration", color="b")
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s^2]")
plt.legend()

plt.tight_layout()
plt.show()

'''
搞了一堆滤波，也是后处理
'''
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt
# from scipy import integrate

# # 定义卡尔曼滤波器类
# class KalmanFilter:
#     def __init__(self, dt, process_noise, measurement_noise):
#         self.dt = dt  # 时间步长

#         # 状态向量 [位置, 速度, 加速度]
#         self.x = np.zeros((3, 1))  # 初始状态为零

#         # 状态转移矩阵 F (匀加速模型)
#         self.F = np.array([[1, dt, 0.5 * dt**2],
#                            [0, 1, dt],
#                            [0, 0, 1]])

#         # 过程噪声协方差矩阵 Q
#         self.Q = process_noise * np.array([[dt**4 / 4, dt**3 / 2, dt**2 / 2],
#                                            [dt**3 / 2, dt**2, dt],
#                                            [dt**2 / 2, dt, 1]])

#         # 测量矩阵 H (只测量位置)
#         self.H = np.array([[1, 0, 0]])

#         # 测量噪声协方差矩阵 R
#         self.R = measurement_noise * np.array([[10]])

#         # 初始误差协方差矩阵 P
#         self.P = np.diag([1, 1, 10])  # 增大加速度误差的初始协方差


#     def predict(self):
#         # 状态预测
#         self.x = np.dot(self.F, self.x)

#         # 误差协方差预测
#         self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

#     def update(self, z):
#         # 计算卡尔曼增益
#         y = z - np.dot(self.H, self.x)  # 测量残差
#         S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # 误差协方差
#         K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # 卡尔曼增益

#         # 更新状态
#         self.x = self.x + np.dot(K, y)

#         # 更新误差协方差
#         I = np.eye(self.P.shape[0])
#         self.P = np.dot(I - np.dot(K, self.H), self.P)

#     def get_state(self):
#         return self.x.copy()  # 返回状态的副本，避免外部修改影响内部状态

# # 模拟非匀加速的真实系统
# def true_system(t):
#     dt = t[1] - t[0] if len(t) > 1 else 0.01  # 使用时间序列的步长
#     acceleration = np.sin(0.5 * t)  # 非匀加速运动
#     velocity = integrate.cumtrapz(acceleration, dx=dt, initial=0)
#     position = integrate.cumtrapz(velocity, dx=dt, initial=0)
#     return position, velocity, acceleration

# # 设计低通 Butterworth 滤波器
# def butter_lowpass(cutoff, fs, order=4):
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     return b, a

# # 双向滤波（Forward-Backward Filtering）
# def bidirectional_filter(data, b, a):
#     return filtfilt(b, a, filtfilt(b, a, data)[::-1])[::-1]

# # 测量噪声和过程噪声的设置
# dt = 0.01  # 时间步长
# process_noise = 1e-2  # 合理的过程噪声
# measurement_noise = 1e-6 # 合理的测量噪声

# # 初始化卡尔曼滤波器
# kf = KalmanFilter(dt, process_noise, measurement_noise)

# # 生成数据
# time_steps = np.arange(0, 10, dt)
# true_positions, true_velocities, true_accelerations = true_system(time_steps)

# measured_positions = []
# estimated_positions = []
# estimated_velocities = []
# estimated_accelerations = []

# # 用于存储实时的加速度估计值
# acceleration_history = []

# for i, t in enumerate(time_steps):
#     # 假设我们只能测量位置，并加入一些噪声
#     measured_pos = true_positions[i] + np.random.normal(0, np.sqrt(measurement_noise))
#     measured_positions.append(measured_pos)

#     # 卡尔曼滤波器的预测和更新步骤
#     kf.predict()
#     kf.update(np.array([[measured_pos]]))

#     # 获取滤波后的估计状态（位置、速度、加速度）
#     estimated_state = kf.get_state()
#     estimated_positions.append(estimated_state[0, 0])
#     estimated_velocities.append(estimated_state[1, 0])
#     estimated_accelerations.append(estimated_state[2, 0])

#     # 存储实时加速度估计值
#     acceleration_history.append(estimated_state[2, 0])

# # 设计低通 Butterworth 滤波器参数
# fs = 1 / dt  # 采样频率
# cutoff = 2  # 截止频率 (Hz)
# b, a = butter_lowpass(cutoff, fs)

# # 对历史加速度估计值进行双向滤波
# smoothed_acceleration = bidirectional_filter(acceleration_history, b, a)

# # 可视化结果
# plt.figure(figsize=(10, 8))

# # 位置对比
# plt.subplot(3, 1, 1)
# plt.plot(time_steps, true_positions, label="True Position", color="g")
# plt.plot(time_steps, measured_positions, label="Measured Position", color="r", linestyle="--")
# plt.plot(time_steps, estimated_positions, label="Estimated Position", color="b")
# plt.xlabel("Time [s]")
# plt.ylabel("Position [m]")
# plt.legend()

# # 速度对比
# plt.subplot(3, 1, 2)
# plt.plot(time_steps, true_velocities, label="True Velocity", color="g")
# plt.plot(time_steps, estimated_velocities, label="Estimated Velocity", color="b")
# plt.xlabel("Time [s]")
# plt.ylabel("Velocity [m/s]")
# plt.legend()

# # 加速度对比
# plt.subplot(3, 1, 3)
# plt.plot(time_steps, true_accelerations, label="True Acceleration", color="g")
# plt.plot(time_steps, smoothed_acceleration, label="Smoothed Acceleration (Bidirectional Filter)", color="r")
# plt.xlabel("Time [s]")
# plt.ylabel("Acceleration [m/s^2]")
# plt.legend()

# plt.tight_layout()
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import integrate
# from scipy.signal import butter, filtfilt

# # 定义卡尔曼滤波器类
# class KalmanFilter:
#     def __init__(self, dt, process_noise, measurement_noise):
#         self.dt = dt  # 时间步长

#         # 状态向量 [位置, 速度, 加速度]
#         self.x = np.zeros((3, 1))  # 初始状态为零

#         # 状态转移矩阵 F (匀加速模型)
#         self.F = np.array([[1, dt, 0.5 * dt**2],
#                            [0, 1, dt],
#                            [0, 0, 1]])

#         # 过程噪声协方差矩阵 Q
#         self.Q = process_noise * np.array([[dt**4 / 4, dt**3 / 2, dt**2 / 2],
#                                            [dt**3 / 2, dt**2, dt],
#                                            [dt**2 / 2, dt, 1]])

#         # 测量矩阵 H (只测量位置)
#         self.H = np.array([[1, 0, 0]])

#         # 测量噪声协方差矩阵 R (设置为一个常数)
#         self.R = measurement_noise * np.array([[10]])

#         # 初始误差协方差矩阵 P
#         self.P = np.diag([1, 1, 10])  # 增大加速度误差的初始协方差


#     def predict(self):
#         # 状态预测
#         self.x = np.dot(self.F, self.x)

#         # 误差协方差预测
#         self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

#     def update(self, z):
#         # 计算卡尔曼增益
#         y = z - np.dot(self.H, self.x)  # 测量残差
#         S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # 误差协方差
#         K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # 卡尔曼增益

#         # 更新状态
#         self.x = self.x + np.dot(K, y)

#         # 更新误差协方差
#         I = np.eye(self.P.shape[0])
#         self.P = np.dot(I - np.dot(K, self.H), self.P)

#     def get_state(self):
#         return self.x.copy()  # 返回状态的副本，避免外部修改影响内部状态

# # 模拟非匀加速的真实系统
# def true_system(t):
#     dt = t[1] - t[0] if len(t) > 1 else 0.01  # 使用时间序列的步长
#     acceleration = np.sin(0.5 * t)  # 非匀加速运动
#     velocity = integrate.cumtrapz(acceleration, dx=dt, initial=0)
#     position = integrate.cumtrapz(velocity, dx=dt, initial=0)
#     return position, velocity, acceleration

# # 双向滤波器
# def bidirectional_filter(data, b, a):
#     # 确保数据量足够进行滤波
#     if len(data) < 2 * max(len(b), len(a)):
#         print(f"Warning: Not enough data to apply bidirectional filter. Data length: {len(data)}")
#         return data  # 如果数据不足，直接返回原数据
#     else:
#         return filtfilt(b, a, filtfilt(b, a, data)[::-1])[::-1]

# # 设置滤波器参数
# fs = 100  # 采样频率（假设 100 Hz）
# cutoff = 2  # 截止频率 (Hz)
# order = 4  # 滤波器阶数
# b, a = butter(order, cutoff / (fs / 2), btype='low')

# # 时间设置
# dt = 0.01  # 时间步长
# process_noise = 1e-2  # 合理的过程噪声
# measurement_noise = 1e-6 # 合理的测量噪声

# # 初始化卡尔曼滤波器
# kf = KalmanFilter(dt, process_noise, measurement_noise)

# # 生成数据
# time_steps = np.arange(0, 10, dt)
# true_positions, true_velocities, true_accelerations = true_system(time_steps)

# measured_positions = []
# estimated_positions = []
# estimated_velocities = []
# estimated_accelerations = []
# acceleration_history = []
# smoothed_accelerations = []

# for i, t in enumerate(time_steps):
#     # 假设我们只能测量位置，并加入一些噪声
#     measured_pos = true_positions[i] + np.random.normal(0, np.sqrt(measurement_noise))
#     measured_positions.append(measured_pos)

#     # 卡尔曼滤波器的预测和更新步骤
#     kf.predict()
#     kf.update(np.array([[measured_pos]]))

#     # 获取滤波后的估计状态（位置、速度、加速度）
#     estimated_state = kf.get_state()
#     estimated_positions.append(estimated_state[0, 0])
#     estimated_velocities.append(estimated_state[1, 0])
#     estimated_accelerations.append(estimated_state[2, 0])

#     # 存储实时加速度估计值
#     acceleration_history.append(estimated_state[2, 0])

#     # 对历史加速度估计值进行双向滤波
#     if len(acceleration_history) > 2 * max(len(b), len(a)):  # 确保有足够的数据
#         smoothed_acceleration = bidirectional_filter(acceleration_history, b, a)
#         smoothed_accelerations.append(smoothed_acceleration[-1])  # 取最后一个元素作为实时估计值
#     else:
#         smoothed_accelerations.append(estimated_state[2, 0])  # 初始时刻直接使用加速度估计

# # 可视化结果
# plt.figure(figsize=(10, 8))

# # 位置对比
# plt.subplot(3, 1, 1)
# plt.plot(time_steps, true_positions, label="True Position", color="g")
# plt.plot(time_steps, measured_positions, label="Measured Position", color="r", linestyle="--")
# plt.plot(time_steps, estimated_positions, label="Estimated Position", color="b")
# plt.xlabel("Time [s]")
# plt.ylabel("Position [m]")
# plt.legend()

# # 速度对比
# plt.subplot(3, 1, 2)
# plt.plot(time_steps, true_velocities, label="True Velocity", color="g")
# plt.plot(time_steps, estimated_velocities, label="Estimated Velocity", color="b")
# plt.xlabel("Time [s]")
# plt.ylabel("Velocity [m/s]")
# plt.legend()

# # 加速度对比
# plt.subplot(3, 1, 3)
# plt.plot(time_steps, true_accelerations, label="True Acceleration", color="g")
# plt.plot(time_steps, smoothed_accelerations, 'r-', label='Smoothed Acceleration (Bidirectional Filter)')

# plt.xlabel("Time [s]")
# plt.ylabel("Acceleration [m/s^2]")
# plt.legend()

# plt.tight_layout()
# plt.show()
