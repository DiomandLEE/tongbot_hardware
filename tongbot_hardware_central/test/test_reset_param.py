import numpy as np
import matplotlib.pyplot as plt
'''
这个当时是看卡尔曼滤波后期有一些误差变大，以为只要把参数重置为一开始就可以解决，结果告诉我是不行的
'''

# 定义卡尔曼滤波器类
class KalmanFilter:
    def __init__(self, dt, process_noise, measurement_noise):
        self.dt = dt  # 时间步长

        # 状态向量 [位置, 速度, 加速度]
        self.x = np.zeros((3, 1))  # 初始状态为零

        # 状态转移矩阵 F (匀加速模型)
        self.F = np.array([[1, dt, 0.5 * dt**2], [0, 1, dt], [0, 0, 1]])

        # 过程噪声协方差矩阵 Q
        self.Q = process_noise * np.array([[dt**4 / 4, dt**3 / 2, dt**2 / 2],
                                           [dt**3 / 2, dt**2, dt],
                                           [dt**2 / 2, dt, 1]])

        # 测量矩阵 H (只测量位置)
        self.H = np.array([[1, 0, 0]])

        # 测量噪声协方差矩阵 R
        self.R = measurement_noise * np.array([[1]])

        # 初始误差协方差矩阵 P
        self.P = np.eye(3)

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
        return self.x

# 模拟非匀加速的真实系统
def true_system(t):
    # 真实的加速度在时间上变化
    acceleration = np.sin(0.5 * t)  # 非匀加速运动
    velocity = acceleration * t  # 速度 = 加速度 * 时间
    position = 0.5 * acceleration * t**2  # 位置 = 0.5 * 加速度 * 时间^2
    return position, velocity, acceleration

# 测量噪声和过程噪声的设置
dt = 0.01  # 时间步长
process_noise = 1e-1  # 过程噪声，适当增加过程噪声
measurement_noise = 1e-1  # 测量噪声，适当增加测量噪声

# 初始化卡尔曼滤波器
kf = KalmanFilter(dt, process_noise, measurement_noise)

# 生成数据
time_steps = np.arange(0, 10, dt)
true_positions = []
measured_positions = []
estimated_positions = []
estimated_velocities = []
estimated_accelerations = []

# 重置卡尔曼滤波器的时间间隔（每1秒重置一次）
reset_interval = 10/2  # 每1秒重置一次

# 初始化上一时刻的估计状态
last_estimated_state = np.zeros((3, 1))

for t in time_steps:
    # 真实系统的状态（非匀加速）
    true_pos, true_vel, true_acc = true_system(t)
    true_positions.append(true_pos)

    # 假设我们只能测量位置，并加入一些噪声
    measured_pos = true_pos + np.random.normal(0, measurement_noise)
    measured_positions.append(measured_pos)

    # 每隔1秒重置卡尔曼滤波器的状态
    if int(t) % reset_interval == 0:
        kf = KalmanFilter(dt, process_noise, measurement_noise)  # 重新初始化卡尔曼滤波器

        # 使用上一时刻的估计结果初始化卡尔曼滤波器
        kf.x = last_estimated_state.copy()  # 使用上一时刻的状态作为新的初始状态
        kf.update(np.array([[measured_pos]]))  # 以当前测量值为初始值进行更新

        # 为了加速适应，可以多进行几次预测和更新
        for _ in range(1000):
            kf.predict()
            kf.update(np.array([[measured_pos]]))

    else:
        # 卡尔曼滤波器的预测和更新步骤
        kf.predict()
        kf.update(np.array([[measured_pos]]))

    # 获取滤波后的估计状态（位置、速度、加速度）
    estimated_state = kf.get_state()
    estimated_positions.append(estimated_state[0, 0])
    estimated_velocities.append(estimated_state[1, 0])
    estimated_accelerations.append(estimated_state[2, 0])

    # 更新上一时刻的估计状态
    last_estimated_state = estimated_state.copy()

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
plt.plot(time_steps, np.gradient(true_positions, dt), label="True Velocity", color="g")
plt.plot(time_steps, estimated_velocities, label="Estimated Velocity", color="b")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend()

# 加速度对比
plt.subplot(3, 1, 3)
plt.plot(time_steps, np.gradient(np.gradient(true_positions, dt), dt), label="True Acceleration", color="g")
plt.plot(time_steps, estimated_accelerations, label="Estimated Acceleration", color="b")
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s^2]")
plt.legend()

plt.tight_layout()
plt.show()
