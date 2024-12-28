import numpy as np
import matplotlib.pyplot as plt
'''
测试ridgeback中的速度指数滤波器和卡尔曼滤波器，哪个更加准确
从结果来看，卡尔曼滤波器的效果更好
但是好像在plot.py中，把指数滤波器视为了真值               
'''

# 定义卡尔曼滤波器
class KalmanFilter:
    def __init__(self, dt):
        self.dt = dt

        # 状态向量 [位置, 速度]
        self.x = np.zeros(2)  # 初始状态为零向量

        # 状态转移矩阵 (匀加速模型)
        self.F = np.array([[1, self.dt],  # 位置更新
                           [0, 1]])       # 速度更新

        # 控制输入矩阵
        self.B = np.array([[0.5 * self.dt**2],  # 加速度对位置的影响
                           [self.dt]])          # 加速度对速度的影响

        # 测量矩阵 (我们只能测量位置)
        self.H = np.array([[1, 0]])  # 只关心位置

        # 过程噪声协方差矩阵
        self.Q = np.array([[0.1, 0.0],  # 位置的噪声
                           [0.0, 0.1]])  # 速度的噪声

        # 测量噪声协方差矩阵
        self.R = np.array([[1]])  # 位置的噪声

        # 初始估计误差协方差矩阵
        self.P = np.eye(2)

    def predict(self, u):
        # 预测步骤
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        # 更新步骤
        y = z - np.dot(self.H, self.x)  # 测量创新
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # 残差协方差
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # 卡尔曼增益
        self.x = self.x + np.dot(K, y)  # 更新状态估计
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)  # 更新估计误差协方差



# 定义指数平滑滤波器
class ExponentialSmoother:
    def __init__(self, tau, x0):
        self.tau = tau
        self.prev = x0

    def next(self, measured, dt):
        c = 1.0 - np.exp(-dt / self.tau)
        state = c * measured + (1 - c) * self.prev
        self.prev = state
        return state

# 生成测试数据：匀加速运动模型
np.random.seed(42)
time = np.linspace(0, 10, 1000)  # 时间 0 到 10 秒
dt = time[1] - time[0]

# 模拟匀加速运动
initial_position = 0
initial_velocity = 0  # 初始速度 2 m/s
acceleration = 0.5  # 加速度 0.5 m/s^2

# 生成理想位置数据
position = initial_position + initial_velocity * time + 0.5 * acceleration * time**2

# 添加噪声
position += 0.005 * np.random.randn(len(position))  # 位置噪声

# 使用卡尔曼滤波器估计位置、速度和加速度
kf = KalmanFilter(dt)

# 存储估计结果
estimated_positions = []
estimated_velocities = []
estimated_accelerations = []

for z in position:
    kf.predict(acceleration)
    kf.update(z)  # 使用位置测量更新卡尔曼滤波器

    # 存储估计的状态
    estimated_positions.append(kf.x[0])
    estimated_velocities.append(kf.x[1])
    # estimated_accelerations.append(kf.x[2])

# 计算实际速度（通过差分）
delta_position = np.diff(position, prepend=position[0])
measured_velocity = delta_position / dt

# 应用滤波器（指数平滑滤波）
linear_velocity_filter = ExponentialSmoother(0.045, 0.0)
angular_velocity_filter = ExponentialSmoother(0.025, 0.0)

filtered_linear_velocities = []
for v in measured_velocity:
    filtered_linear_velocities.append(linear_velocity_filter.next(v, dt))

filtered_linear_velocities = np.array(filtered_linear_velocities)

# 可视化结果
plt.figure(figsize=(10, 6))

# 绘制不同速度的对比图
plt.plot(time, measured_velocity, label="Measured Velocity", alpha=0.6, color="orange")
plt.plot(time, filtered_linear_velocities, label="Filtered Velocity (Exponential)", color="blue")
plt.plot(time, estimated_velocities, label="Estimated Velocity (Kalman)", color="green")
plt.title("Velocity Comparison (Kalman vs Exponential Filter)")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend()
plt.grid()
plt.show()
