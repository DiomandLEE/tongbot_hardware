import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.signal import butter, filtfilt

'''
还是一样测试对小车的状态估计，只是之前忘记去除双向滤波了，现在去掉了，效果一样
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

        # 输入矩阵 B (与 jerk 相关)
        self.B = np.array([[dt**3 / 6],
                           [dt**2 / 2],
                           [dt]])

        # 过程噪声协方差矩阵 Q (w * B * B^T)
        self.Q = process_noise * np.dot(self.B, self.B.T)

        # 测量矩阵 H (只测量位置)
        self.H = np.array([[1, 0, 0]])

        # 测量噪声协方差矩阵 R (设置为一个常数)
        self.R = measurement_noise * np.array([[10]])

        # 初始误差协方差矩阵 P
        self.P = np.diag([1, 1, 10])  # 增大加速度误差的初始协方差

    def predict(self, u):
        # 状态预测
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)

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

# 模拟带有 jerk 输入的真实系统
def true_system(t):
    dt = t[1] - t[0] if len(t) > 1 else 0.01  # 使用时间序列的步长

    # 加权叠加多个三角函数来生成加速度（代表 jerk 输入）
    freq1, amp1, phase1 = 0.5, 1.0, 0  # 第一个三角函数
    freq2, amp2, phase2 = 1.0, 0.5, np.pi / 4  # 第二个三角函数
    freq3, amp3, phase3 = 2.0, 0.2, np.pi / 2  # 第三个三角函数

    # 生成加速度（多个三角函数的叠加）
    acceleration = (
        amp1 * np.sin(freq1 * t + phase1) +
        amp2 * np.cos(freq2 * t + phase2) +
        amp3 * np.sin(freq3 * t + phase3)
    )

    # 计算速度和位置（累积积分）
    velocity = integrate.cumtrapz(acceleration, dx=dt, initial=0)
    position = integrate.cumtrapz(velocity, dx=dt, initial=0)

    # 计算 jerk 输入（加加速度，作为输入）
    jerk = np.gradient(acceleration, dt)

    return position, velocity, acceleration, jerk

# 时间设置
dt = 0.01  # 时间步长
process_noise = 1e-3  # 合理的过程噪声
measurement_noise = 1e-6 # 合理的测量噪声

# 初始化卡尔曼滤波器
kf = KalmanFilter(dt, process_noise, measurement_noise)

# 生成数据
time_steps = np.arange(0, 10, dt)
true_positions, true_velocities, true_accelerations, jerk_input = true_system(time_steps)

measured_positions = []
estimated_positions = []
estimated_velocities = []
estimated_accelerations = []

# 开始卡尔曼滤波和加速度估计
for i, t in enumerate(time_steps):
    # 假设我们只能测量位置，并加入一些噪声
    measured_pos = true_positions[i] + np.random.normal(0, np.sqrt(measurement_noise))#* 100000))
    measured_positions.append(measured_pos)

    # 卡尔曼滤波器的预测和更新步骤
    kf.predict(jerk_input[i])  # 将 jerk 输入作为控制输入
    kf.update(np.array([[measured_pos]]))

    # 获取滤波后的估计状态（位置、速度、加速度）
    estimated_state = kf.get_state()
    estimated_positions.append(estimated_state[0, 0])
    estimated_velocities.append(estimated_state[1, 0])
    estimated_accelerations.append(estimated_state[2, 0])

# 计算误差
position_error = np.array(estimated_positions) - true_positions
velocity_error = np.array(estimated_velocities) - true_velocities
acceleration_error = np.array(estimated_accelerations) - true_accelerations

# 可视化结果
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
plt.plot(time_steps[100:], estimated_velocities[100:], label="Estimated Velocity", color="b")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend()

# 加速度对比
plt.subplot(4, 1, 3)
plt.plot(time_steps[100:], true_accelerations[100:], label="True Acceleration", color="g")
plt.plot(time_steps[100:], estimated_accelerations[100:], label="Estimated Acceleration", color="b")
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s^2]")
plt.legend()

# 加速度误差
plt.subplot(4, 1, 4)
plt.plot(time_steps[100:], acceleration_error[100:], label="Acceleration Error", color="r")
plt.xlabel("Time [s]")
plt.ylabel("Error [m/s^2]")
plt.legend()

plt.tight_layout()
plt.show()
