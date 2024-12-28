# collect_base_calibration_data.py
"""
收集底座的校准数据，用于调整旋转中心的位置。

程序通过将底座移动到一系列目标配置，记录每个点的平均测量配置，最终保存这些数据。
方法：
找到marker距离base中点的x和y方向的距离，让base旋转，
如果marker和base中心重合的话，那么maker点的x和y方向是不会有变化的
"""

# 导入必要的库
import argparse  # 用于解析命令行参数
import datetime  # 用于获取当前时间戳
import rospy  # 用于ROS节点的创建和通信
import numpy as np  # 用于数值计算
import tongbot_hardware_central as tongbot  # 假设这是一个机器人相关的库

# 定义一些常量
MAX_JOINT_VELOCITY = 0.2  # 最大关节速度（m/s 或 rad/s）
P_GAIN = 0.5  # 比例增益，用于控制
CONVERGENCE_TOL = 1e-2  # 收敛容差，判断是否达到目标
RATE = 100  # 控制循环频率（Hz）

# 定义一个函数，用于计算某一时间段内的平均配置


def average_configuration(robot, rate, duration=5.0):
    """
    在指定的持续时间内，计算机器人配置的平均值。

    参数:
    - robot: 机器人接口对象
    - rate: ROS循环频率对象
    - duration: 持续时间（秒）

    返回:
    - 平均配置 (numpy 数组)
    """
    qs = []  # 用于存储每次测量的配置
    t0 = rospy.Time.now().to_sec()  # 记录初始时间
    t = t0
    while not rospy.is_shutdown() and t - t0 < duration:  # 持续测量直到超时或ROS关闭
        qs.append(robot.q.copy())  # 读取当前配置
        rate.sleep()  # 等待下一个周期
        t = rospy.Time.now().to_sec()  # 更新当前时间
    return np.mean(qs, axis=0)  # 返回测量配置的平均值

# 主函数


def main():
    """
    主程序，负责运行校准数据采集流程。
    """
    # 定义命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        help="Filename for the saved data. Timestamp is automatically appended.", # 保存数据的文件名，程序会自动添加时间戳。
        nargs="?",
        default="base_calibration_data",
    )
    args = parser.parse_args()

    # 初始化ROS节点
    rospy.init_node("base_calibration_data_collection")

    # 获取当前时间戳，用于保存文件时标记
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    robot = tongbot.DingoROSInterface()  # 创建机器人接口对象
    rate = rospy.Rate(RATE)  # 创建ROS循环频率对象

    # 等待机器人反馈（确保机器人准备好）
    while not rospy.is_shutdown() and not robot.ready():
        rate.sleep()

    # 初始化目标配置，仅旋转底座
    q0 = robot.q.copy()  # 获取机器人当前配置
    θds = np.array([0, np.pi / 6, np.pi / 3, np.pi / 2, 0]) + q0[2]  # 目标角度序列，逐渐转90度，再转回来
    num_configs = θds.shape[0]  # 配置数量

    # 用于存储测量配置的列表
    qs = []

    # 仅控制角度，用于校准位置
    idx = 0
    while not rospy.is_shutdown():
        error = θds[idx] - robot.q[2]  # 计算当前配置与目标角度的误差
        if np.abs(error) < CONVERGENCE_TOL:  # 如果误差小于容差，认为已到达目标
            robot.brake()  # 停止底座
            print(f"Converged to location {idx}.")  # 打印当前配置索引

            idx += 1  # 准备进入下一个目标配置
            if idx >= num_configs:  # 如果所有目标配置已完成，退出循环
                break

            q = average_configuration(robot, rate)  # 获取当前配置的平均值
            qs.append(q)  # 保存配置
            print(f"Average configuration = {q}.")  # 打印平均配置

        # 根据误差发布速度命令，仅调整角度
        cmd_vel = np.array([0, 0, P_GAIN * error])  # 计算速度命令
        cmd_vel = tongbot.bound_array(
            cmd_vel, lb=-MAX_JOINT_VELOCITY, ub=MAX_JOINT_VELOCITY)  # 限制速度范围
        robot.publish_cmd_vel(cmd_vel)  # 发布速度命令

        rate.sleep()  # 等待下一个周期

    robot.brake()  # 停止机器人

    # 保存采集的数据
    filename = f"{args.filename}_{timestamp}.npz"  # 拼接文件名
    np.savez_compressed(filename, q0=q0, θds=θds,
                        qs=qs)  # 保存为压缩格式，将多个数组压缩成一个文件内
    print(f"Base calibration data saved to {filename}.")  # 打印保存路径


# 主程序入口
if __name__ == "__main__":
    main()
