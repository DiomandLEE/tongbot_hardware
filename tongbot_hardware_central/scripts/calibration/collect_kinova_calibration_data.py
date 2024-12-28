#!/usr/bin/env python3
"""收集机械臂的标定数据。

将机械臂移动到一系列预定的配置点，记录每个点的平均测量配置，并保存数据。

请注意，基座也必须运行，因为其（静止的）姿态也由 Vicon 记录。
base是静止的，这对整体的标定是不影响的
"""
import argparse
import datetime
import time

import rospy
import numpy as np
from scipy.spatial.transform import Rotation

import tongbot_hardware_central as tongbot

# 设置常量
MAX_JOINT_VELOCITY = 0.2  # 最大关节速度（单位：rad/s）
MAX_JOINT_ACCELERATION = 0.3  # 最大关节加速度（单位：rad/s^2）
P_GAIN = 1  # 比例增益
CONVERGENCE_TOL = 1e-3  # 收敛容忍度
RATE = 100  # 数据收集频率（单位：Hz），125更改为100hz
EE_OBJECT_NAME = "camera_link"  # 末端执行器物体名称,标定的时候把marker放在camera的上、左、右

# fmt: off
# 下面的代码块不会被自动格式化
# 最后的机械臂配置与第一个相同，标定完成后机械臂会返回初始位置
# 预定的机械臂配置点
# done  todo 这个需要download UR5e的配置文件，然后在rVIz中看一下这些的关节构型，而后搞出一个kinova的标定中间点
DESIRED_KINOVA_CONFIGURATIONS = np.array([
  [1.5708, 0.7854,  0.35,  1.5708,  -1.35,  -0.35,  0.7854],
  [0,      0.7854,  0.35,  1.5708,  -1.35,  -0.35,  0.7854],
  [2.3562, 0.7854,  0.35,  1.5708,  -1.35,  -0.35,  0.7854],
  [2.3562, -0.100,  0.35,  1.5708,  -1.35,  -0.35,  0.7854],
  [2.3562, -0.100,  0.35,  0.7854,  -1.35,  -0.35,  0.7854],
  [2.3562, -0.100,  0.35,  0.7854,   -0.5,  0.5708, 0.7854],
  [2.3562, -0.100,  -0.2,  0.7854,  -0.45,  0.5708, 0.7854],
  [1.5708, -0.100,  -0.2,  0.7854,  -0.45,  0.5708, 0.0000],
  [0,      -0.100,  -0.2,  0.7854,  -0.45,  0.5708, 0.0000],
  [1.5708, 0.7854,  0.35,  1.5708,  -1.35,  -0.35,  0.7854]])
# fmt: on

#! 对四元数求平均值


def average_quaternion(Qs):
    """计算一组四元数的平均值。

    四元数的顺序为``[x, y, z, w]``。

    参数
    ----------
    Qs : np.ndarray, shape (n, 4)
        要平均的四元数。

    返回
    -------
    : np.ndarray, shape (4,)
        平均四元数。
    """
    Qs = np.array(Qs)
    e, V = np.linalg.eig(Qs.T @ Qs)  # 计算特征值和特征向量
    i = np.argmax(e)  # 找到最大特征值对应的特征向量
    Q_avg = V[:, i]  # 得到平均四元数

    # TODO 可以使用 scipy，但为了使用标准方法，要求安装较新的版本
    # more recent version
    # Q_avg2 = Rotation.from_quat(Qs).mean().as_quat()
    # print(f"Q_avg = {Q_avg}")
    # print(f"Q_avg2 = {Q_avg2}")
    return Q_avg

#! 对测量值求平均，包括关节配置、物体位置和物体姿态角
# 在一个位置停住，而后取一段时间内的平均值


def average_measurements(robot, vicon, rate, duration=5.0):
    """计算给定持续时间内的平均测量值。

    参数
    ----------
    robot : MobileManipulatorROSInterface
        机械臂接口。
    vicon : ViconObjectInterface
        Vicon对象接口。
    rate : rospy.Rate
        测量循环的频率。
    duration : float, 非负
        用于计算平均值的持续时间。

    返回
    -------
    : tuple
        返回一个元组 ``(q, r, Q)``, 其中 ``q`` 是平均的关节配置，
        ``r`` 是平均的物体位置，``Q`` 是平均的物体姿态（四元数 ``[x, y, z, w]``）。
    """
    qs = []  # 关节配置列表
    Qs = []  # object姿态四元数列表
    rs = []  # object位移列表

    t0 = rospy.Time.now().to_sec()  # 获取当前时间
    t = t0
    while not rospy.is_shutdown() and t - t0 < duration:  # 循环收集数据，直到持续时间结束
        qs.append(robot.q.copy())  # 保存机械臂的关节配置
        rs.append(vicon.position.copy())  # 保存物体的位置
        Qs.append(vicon.orientation.copy())  # 保存物体的姿态

        rate.sleep()  # 根据设定频率休眠
        t = rospy.Time.now().to_sec()

    # 计算平均值
    q = np.mean(qs, axis=0)  # 关节配置的平均值
    r = np.mean(rs, axis=0)  # 物体位置的平均值
    Q = average_quaternion(Qs)  # 物体姿态四元数的平均值
    return q, r, Q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        help="Filename for the saved data. Timestamp is automatically appended.", #保存数据的文件名。时间戳会自动附加。
        nargs="?",
        default="arm_calibration_data",
    )
    parser.add_argument(
        "--home-config",
        help="Path to YAML file to load the home configurations from.", # 加载home位置配置的YAML文件路径。
        required=True,
    )
    parser.add_argument(
        "--home-name",
        help="Name of the home position to use for the base.", # 用于基座的家位置名称。
        default="default",
    )
    # 执行上述代码的时候，终端需要加载参数：
    # python script_name.py my_custom_filename --home-config /path/to/home_config.yaml --home-name custom_home
    args = parser.parse_args()

    # 加载家位置，用于确定基座在数据收集过程中应停留的位置
    home = tongbot.load_home_position(name=args.home_name, path=args.home_config)

    rospy.init_node("dingo_kinova_calibration_data_collection", disable_signals=True)
    signal_handler = tongbot.SimpleSignalHandler()

    # 使用时间戳作为文件名的一部分
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    robot = tongbot.MobileManipulatorROSInterface()  # 初始化机器人接口 #todo 这里也要修改为kinova的
    vicon = tongbot.ViconObjectInterface(EE_OBJECT_NAME)  # 初始化Vicon接口
    rate = rospy.Rate(RATE)  # 设置数据收集频率

    # 等待直到机器人和Vicon反馈都准备好
    while not rospy.is_shutdown() and not signal_handler.received:  # SimpleSignalHandler，在中断信号前，对类中的flag进行处理
        if robot.ready() and vicon.ready():
            break
        rate.sleep()

    q0 = robot.q.copy()  # 获取初始的关节配置 #! 这里的robot就是base+arm，不单是文件名说的那样只是arm

    num_configs = DESIRED_KINOVA_CONFIGURATIONS.shape[0]  # 预定配置点的数量，这个只对arm
    goals = np.hstack(
        (
            np.tile(home[:3], (num_configs, 1)),  # 将home位置扩展为与目标配置相同的维度
            DESIRED_KINOVA_CONFIGURATIONS,  # 预定的机械臂配置
        )
    )  # ! 把home的前三个位置扩展为与ARM目标配置相同的维度(行数)，然后与目标配置拼接在一起，得到完整的配置点

    # 记录测量的配置（不完全与预定相同）
    qs = []

    # 记录Vicon的位姿（包括位置和姿态）
    rs = []
    Qs = []

    # 设置第一个目标
    goal = goals[0, :]  # 第一行的所有数据，即base+arm的第一个配置点
    delta = goal - q0  # 计算起始点到目标点的差值
    trajectory = tongbot.PointToPointTrajectory.quintic(
        start=q0,
        delta=delta,
        max_vel=MAX_JOINT_VELOCITY,
        max_acc=MAX_JOINT_ACCELERATION,
    )  # !得到每一个关节的五次多项式轨迹
    print(f"Moving to goal 0 with duration {trajectory.duration} seconds.")

    # 使用比例控制（P控制）导航到目标位置，并限制速度
    idx = 0
    while not rospy.is_shutdown() and not signal_handler.received:
        t = rospy.Time.now().to_sec()
        dist = np.linalg.norm(goal - robot.q)  # 计算当前配置与目标配置的距离
        if trajectory.done(t) and dist < CONVERGENCE_TOL:  # 如果轨迹完成并且距离足够小
            robot.brake()  # 停止机器人
            print(f"Converged to location {idx} with distance {dist}.")

            idx += 1  # idx是目标配置的个数，和五次多项式无关
            if idx >= num_configs:  # 如果所有目标配置都已完成，退出循环
                break

            # 计算当前配置的平均值
            q, r, Q = average_measurements(robot, vicon, rate)
            qs.append(q)
            rs.append(r)
            Qs.append(Q)

            print(f"Average configuration = {q}.")
            print(f"Average position = {r}.")
            print(f"Average quaternion = {Q}.")
            #! 到达一个目标配置之后，就开始计算当前的状态，并构建下一个

            # 为下一个目标配置构建轨迹
            goal = goals[idx, :]
            delta = goal - q
            trajectory = tongbot.PointToPointTrajectory.quintic(
                start=q,
                delta=delta,
                max_vel=MAX_JOINT_VELOCITY,
                max_acc=MAX_JOINT_ACCELERATION,
            )
            print(
                f"Moving to goal {idx} with duration {trajectory.duration} seconds.")

        t = rospy.Time.now().to_sec()
        qd, vd, _ = trajectory.sample(t)  # 采样轨迹的期望配置
        cmd_vel = P_GAIN * (qd - robot.q) + vd  # 使用P控制生成速度命令 #! 前馈pid
        cmd_vel = tongbot.bound_array(
            cmd_vel, lb=-MAX_JOINT_VELOCITY, ub=MAX_JOINT_VELOCITY)  # 限制速度
        robot.publish_cmd_vel(cmd_vel)  # 发布速度命令

        rate.sleep()

    print("Braking robot")
    robot.brake()  # 停止机器人
    time.sleep(1.0)

    # read 以上就是收集完数据了，但是有几个地方有些疑惑，为什么要有速度控制来移动到目标位置，而不是位置控制移动到那里，毕竟base是不变的
    # just do it,就是这样做的，不用纠结

    # 将数据保存到文件
    filename = f"{args.filename}_{timestamp}.npz"
    print(f"Saving data to {filename}...")
    np.savez(
        filename,
        q=qs,
        r=rs,
        Q=Qs,
    )

    print("Done.")


if __name__ == "__main__":
    main()
