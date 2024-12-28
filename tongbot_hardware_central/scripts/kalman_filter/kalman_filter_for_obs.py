import rospy
import numpy as np
from sensor_msgs.msg import JointState
from scipy.signal import butter, filtfilt
import tf
from geometry_msgs.msg import PoseStamped
from scipy import integrate
from tf import TransformListener

# 定义卡尔曼滤波器类
class DynamicObsKalmanFilter:
    def __init__(self, dt, process_noise, measurement_noise_x, measurement_noise_y, filt_a, filt_b, bidirectional_filter_flag=False):
        self.dt = dt  # 时间步长
        self.bidirectional_filter_flag = bidirectional_filter_flag  # 是否启用双向滤波

        # 状态向量 [x位置, y位置, x速度, y速度, x加速度, y加速度]
        self.x = np.zeros((6, 1))  # 初始状态为零

        # 状态转移矩阵 F (匀加速模型)
        self.F = np.array([[1, 0, dt, 0, 0.5 * dt**2, 0],
                           [0, 1, 0, dt, 0, 0.5 * dt**2],
                           [0, 0, 1, 0, dt, 0],
                           [0, 0, 0, 1, 0, dt],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])

        # 分别为x和y方向的过程噪声协方差矩阵 Q
        self.Q = process_noise * np.array([[self.dt**4 / 4, 0, self.dt**3 / 2, 0, self.dt**2 / 2, 0],
                                           [0, self.dt**4 / 4, 0, self.dt**3 / 2, 0, self.dt**2 / 2],
                                           [self.dt**3 / 2, 0, self.dt**2, 0, self.dt, 0],
                                           [0, self.dt**3 / 2, 0, self.dt**2, 0, self.dt],
                                           [self.dt**2 / 2, 0, self.dt, 0, 1, 0],
                                           [0, self.dt**2 / 2, 0, self.dt, 0, 1]])

        # 测量矩阵 H (只测量位置)
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]])

        # 分别为x和y方向的测量噪声协方差矩阵 R
        self.R_x = measurement_noise_x * np.array([[10]])
        self.R_y = measurement_noise_y * np.array([[10]])  # 同样可以根据实际需求调整

        # 初始误差协方差矩阵 P
        self.P = np.diag([1, 1, 1, 10, 10, 10])  # 增大加速度误差的初始协方差

        self.acceleration_history_x = []
        self.acceleration_history_y = []

        self.a = filt_a
        self.b = filt_b

        self.measure_num = 0

    def predict(self):
        # 状态预测
        self.x = np.dot(self.F, self.x)

        # 误差协方差预测
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        # 计算卡尔曼增益
        y = z - np.dot(self.H, self.x)  # 测量残差
        S = np.dot(np.dot(self.H, self.P), self.H.T) + np.block([[self.R_x, np.zeros((1, 1))], [np.zeros((1, 1)), self.R_y]])  # 误差协方差
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # 卡尔曼增益

        # 更新状态
        self.x = self.x + np.dot(K, y)

        # 更新误差协方差
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_state(self):
        return self.x.copy()  # 返回状态的副本，避免外部修改影响内部状态

    # 双向滤波器
    def bidirectional_filter(self, data):
        # 确保数据量足够进行滤波
        min_data_length = 2 * max(len(self.b), len(self.a))  # 双向滤波器所需的最小数据量
        if len(data) < min_data_length:
            rospy.logwarn(f"Not enough data to apply bidirectional filter. Data length: {len(data)}")
            return data  # 如果数据不足，直接返回原数据
        else:
            # 对数据进行双向滤波
            try:
                return filtfilt(self.b, self.a, filtfilt(self.b, self.a, data)[::-1])[::-1]
            except ValueError as e:
                rospy.logerr(f"Error applying bidirectional filter: {e}")
                return data

    def acceleration_filter(self, data_x, data_y):
        # 确保数据量足够进行滤波
        flag = len(data_x) >= 2 * max(len(self.b), len(self.a)) and len(data_y) >= 2 * max(len(self.b), len(self.a))
        if flag:
            smoothed_acceleration_x = self.bidirectional_filter(data_x)
            smoothed_acceleration_y = self.bidirectional_filter(data_y)

            return smoothed_acceleration_x, smoothed_acceleration_y
        else:
            return data_x, data_y


def vicon_callback(msg, kf, flag_filter):
    # 提取Vicon数据  /tf类型
    measured_pos_x = msg.transform.translation.x
    measured_pos_y = msg.transform.translation.y

    kf.measure_num += 1

    # 卡尔曼滤波器的预测和更新步骤
    kf.predict()
    kf.update(np.array([[measured_pos_x], [measured_pos_y]]))

    # 获取滤波后的估计状态（位置、速度、加速度）
    estimated_state = kf.get_state()

    # 估计位置、速度和加速度
    estimated_position_x = estimated_state[0, 0]
    estimated_position_y = estimated_state[1, 0]
    estimated_velocity_x = estimated_state[2, 0]
    estimated_velocity_y = estimated_state[3, 0]


    estimated_acceleration_x = estimated_state[4, 0]
    estimated_acceleration_y = estimated_state[5, 0]

    kf.acceleration_history_x.append(estimated_state[4, 0])
    kf.acceleration_history_y.append(estimated_state[5, 0])

    # 如果启用了双向滤波
    if flag_filter:
        smoothed_acceleration_x, smoothed_acceleration_y = kf.bidirectional_filter(kf.acceleration_history_x, kf.acceleration_history_y)

        kf.acceleration_history_x.pop()
        kf.acceleration_history_y.pop()
        kf.acceleration_history_x.append(smoothed_acceleration_x[-1])
        kf.acceleration_history_y.append(smoothed_acceleration_y[-1])

        estimated_acceleration_x = smoothed_acceleration_x[-1]
        estimated_acceleration_y = smoothed_acceleration_y[-1]


    # 创建JointState消息并发布
    joint_state_msg = JointState()
    joint_state_msg.header.stamp = rospy.Time.now()
    joint_state_msg.name = ['dynamic_x', 'dynamic_y']
    joint_state_msg.position = [estimated_position_x, estimated_position_y]
    joint_state_msg.velocity = [estimated_velocity_x, estimated_velocity_y]
    joint_state_msg.effort = [estimated_acceleration_x, estimated_acceleration_y]

    joint_state_pub.publish(joint_state_msg)

def dynamic_obs_kf_node():
    rospy.init_node('kalman_filter_node', anonymous=True)

     # 设置滤波器参数
    fs = 100  # 采样频率（假设 100 Hz）
    cutoff = 15  # 截止频率 (Hz)
    order = 5  # 滤波器阶数
    global b, a
    b, a = butter(order, cutoff / (fs / 2), btype='low')

    # 获取是否启用双向滤波的flag
    flag_filter = rospy.get_param('/dynamic_obs/use_bidirectional_filter', False)

    # 初始化卡尔曼滤波器
    dt = 0.01  # 时间步长
    process_noise = 1e-3  # 合理的过程噪声
    measurement_noise = 1e-4  # 合理的测量噪声
    kf = DynamicObsKalmanFilter(dt, process_noise, measurement_noise, measurement_noise, a, b, flag_filter)

    # 订阅Vicon数据
    rospy.Subscriber("/vicon/object", PoseStamped, vicon_callback, callback_args=(kf, flag_filter)) #绑定额外的参数
    #! 要注意更改/vicon发布的频率，与dt对应上

    # 发布动态状态
    global joint_state_pub
    joint_state_pub = rospy.Publisher('/dynamic_obs/joint_states', JointState, queue_size=2)

    rospy.spin()

if __name__ == '__main__':
    dynamic_obs_kf_node()
