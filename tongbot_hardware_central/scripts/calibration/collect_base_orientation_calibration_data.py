#!/usr/bin/env python3
"""
收集底座的校准数据，用于调整旋转中心的位置。

程序会将底座移动到一系列目标位置，在每个位置采集平均配置，并保存数据。
方法：
让底盘沿者x轴方向移动，由于要是vicon上base的x轴与world系上的x轴是重合的话，是没有偏移的
若是没有偏移的话，那么在是不会在y方向上有变化的
"""

# 导入必要的库
import argparse  # 用于解析命令行参数
import datetime  # 用于生成文件的时间戳
import rospy  # ROS 的 Python 接口，用于与机器人交互
import numpy as np  # 用于数值计算
import tongbot_hardware_central as tongbot  # 这是一个机器人控制的库

# 定义一些常量
MAX_JOINT_VELOCITY = 0.2  # 最大关节速度（m/s 或 rad/s），这里把所有的可移动都视为joint
P_GAIN = 0.5  # 比例增益，用于控制器的比例部分
CONVERGENCE_TOL = 1e-2  # 收敛容差，当误差小于该值时认为达到目标
RATE = 100  # 控制循环频率（每秒的循环次数）

# 定义一个函数，用于计算机器人配置的平均值


def average_configuration(robot, rate, duration=5.0):
    """
    在指定时间内计算机器人配置的平均值。

    参数:
    - robot: 机器人接口对象
    - rate: ROS 循环频率对象
    - duration: 持续时间（秒）

    返回:
    - 平均配置 (numpy 数组)
    """
    qs = []  # 用于存储测量的配置
    t0 = rospy.Time.now().to_sec()  # 记录当前时间
    t = t0
    while not rospy.is_shutdown() and t - t0 < duration:  # 在指定时间内不断采集配置
        qs.append(robot.q.copy())  # 将当前配置添加到列表中
        rate.sleep()  # 等待下一个周期
        t = rospy.Time.now().to_sec()  # 更新时间
    return np.mean(qs, axis=0)  # 计算所有配置的平均值并返回

# 定义主函数


def main():
    """
    主程序：运行底座校准流程。
    """
    # 定义命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        help="Filename for the saved data. Timestamp is automatically appended.", # 保存数据的文件名（程序会自动添加时间戳）。
        nargs="?",
        default="base_orientation_calibration_data",
    )
    args = parser.parse_args()

    # 初始化 ROS 节点
    rospy.init_node("base_calibration_data_collection")

    # 获取当前时间戳，后续用于文件命名
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 初始化机器人接口对象
    robot = tongbot.DingoROSInterface() #todo 更改ros_interface，注意再看一下，interface
    # 中有没有spin()这个main里是没有的
    rate = rospy.Rate(RATE)  # 定义 ROS 循环频率

    # 等待机器人准备好
    while not rospy.is_shutdown() and not robot.ready():
        rate.sleep()

    # 定义目标配置：仅沿 x 方向移动
    q0 = robot.q.copy()  # 获取机器人当前配置
    # 定义目标位置序列，这一共是四个位置，+q0[0] 是为了使目标位置相对于当前配置
    xds = np.array([0, 1.0, 2.0, 0]) + q0[0]
    num_configs = xds.shape[0]  # 目标配置的数量

    # 初始化测量配置列表
    qs = []

    # 控制循环：移动机器人并记录配置
    idx = 0
    while not rospy.is_shutdown():
        error = xds[idx] - robot.q[0]  # 计算当前配置与目标位置的误差
        if np.abs(error) < CONVERGENCE_TOL:  # 如果误差小于容差，认为已到达目标
            robot.brake()  # 停止机器人
            print(f"Converged to location {idx}.")  # 打印当前目标序号

            idx += 1  # 进入下一个目标
            if idx >= num_configs:  # 如果所有目标完成，退出循环
                break

            q = average_configuration(robot, rate)  # 获取当前位置的平均配置
            qs.append(q)  # 保存到列表中
            print(f"Average configuration = {q}.")  # 打印平均配置

        # 计算速度命令，仅控制 x 方向
        cmd_vel = np.array([P_GAIN * error, 0, 0])  # 生成速度命令
        cmd_vel = tongbot.bound_array(
            cmd_vel, lb=-MAX_JOINT_VELOCITY, ub=MAX_JOINT_VELOCITY)  # 限制速度范围
        robot.publish_cmd_vel(cmd_vel)  # 发布速度命令

        rate.sleep()  # 等待下一个周期

    robot.brake()  # 停止机器人

    # 保存采集的数据
    filename = f"{args.filename}_{timestamp}.npz"  # 拼接文件名
    np.savez_compressed(filename, q0=q0, xds=xds, qs=qs)  # 保存数据为压缩文件
    print(f"Base calibration data saved to {filename}.")  # 打印保存路径


# 程序入口
if __name__ == "__main__":
    main()
