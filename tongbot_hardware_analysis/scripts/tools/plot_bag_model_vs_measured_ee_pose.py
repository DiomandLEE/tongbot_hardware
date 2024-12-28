#!/usr/bin/env python3
"""Plot pose of wood tray as measured by Vicon compared to the pose from the model."""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt
import IPython

from tongbot_hardware_central import ros_utils

import tongbot_hardware_analysis as tanalysis
from tongbot_hardware_central import MobileManipulatorKinematics
# 导入了项目中的三个模块 ros_utils、upright_core、upright_control 和 upright_cmd，提供特定功能（如配置加载、控制模型解析等）

'''
就是针对运动学计算的ee pose，和vicon下测量的ee pose进行了z轴的角度对比，以及旋转的四元数误差对比。
因为upright是保持平衡，所以对于z轴的角度和ee姿态是很重要的。
对于我们自己的话，就是考察位置为主
'''
#! 终端执行： python3 script_name.py --bag-dir /path/to

# def parse_control_model(config_path):
#     config = core.parsing.load_config(config_path)
#     ctrl_config = config["controller"]
#     return ctrl.manager.ControllerModel.from_config(ctrl_config)

# 加载配置文件并提取控制器相关配置。
# core.parsing.load_config: 加载配置文件。
# ctrl.manager.ControllerModel.from_config: 根据配置创建控制模型。
# 这个需要再编译完成之后，看一下能不能运行成功


def main():
    np.set_printoptions(precision=3, suppress=True)
    # 设置numpy的打印精度（小数点后3位，不以科学计数法显示）。

    # parse CLI args (directory containing bag and config file)
    # 解析命令行参数（包含bag文件和配置文件的目录）
    parser = argparse.ArgumentParser()
    tanalysis.cli.add_bag_dir_arguments(parser)
    config_path, bag_path = tanalysis.cli.parse_bag_dir_args(parser.parse_args())

    # model = parse_control_model(config_path)
    # robot = model.robot
    # # 根据配置路径解析控制模型，并获取对应的机器人模型。

    robot = MobileManipulatorKinematics() #read 是要在这里面，修改robot urdf的文件位置

    bag = rosbag.Bag(bag_path)

    kinova_msgs = [msg for _, msg, _ in bag.read_messages("/my_gen3/joint_states")]
    dingo_msgs = [msg for _, msg,
                      _ in bag.read_messages("/dingo/joint_states")]
    tray_msgs = [
        msg for _, msg, _ in bag.read_messages("/vicon/ThingWoodTray/ThingWoodTray")
    ] #todo 修改为需要测试的ee_link，我们采取robotiq的base_link
    # 打开bag文件并读取以下话题的数据：
    #     /my_gen3/joint_states: UR10机械臂关节状态。
    #     /dingo/joint_states: Ridgeback移动底盘关节状态。
    #     /vicon/ThingWoodTray/ThingWoodTray: Vicon测量的托盘位姿。

    ts = ros_utils.parse_time(tray_msgs, normalize_time=False)
    kinova_ts, kinova_qs, _ = ros_utils.parse_kinova_joint_state_msgs(
        kinova_msgs, normalize_time=False
    )
    rb_ts, rb_qs, _ = ros_utils.parse_dingo_joint_state_msgs(
        dingo_msgs, normalize_time=False
    )

    kinova_qs_aligned = ros_utils.interpolate_list(ts, kinova_ts, kinova_qs)
    rb_qs_aligned = ros_utils.interpolate_list(ts, rb_ts, rb_qs)
    # 解析时间戳，arm和base的状态，与插值
    qs = np.hstack((rb_qs_aligned, kinova_qs_aligned))  # 拼成一个长向量
    n = qs.shape[0]

    # prepend default obstacle positions, which we don't care about
    # 在前面，添加默认的障碍物位置，这些位置我们并不关心
    qs = np.hstack((np.zeros((n, 3)), qs))
    ts -= ts[0]  # 时间戳归一化（从0开始）。

    # # just for comparison
    # q_home = model.settings.initial_state[:9]
    # q_home = np.concatenate((np.zeros(3), q_home))
    # robot.forward_qva(q_home)
    # r_home, Q_home = robot.link_pose()

    # compute modelled EE poses
    z = np.array([0, 0, 1])
    ee_positions = np.zeros((n, 3))
    ee_orientations = np.zeros((n, 4))
    ee_angles = np.zeros(n)
    for i in range(n):
        robot.forward_qva(qs[i, :])  # 更新了之后的qs，包含了障碍物的位置在前面
        ee_positions[i, :], ee_orientations[i, :] = robot.link_pose()  # 前向运动学
        R = tanalysis.math.quat_to_rot(ee_orientations[i, :])

        # angle from the upright direction
        ee_angles[i] = np.arccos(z @ R @ z)  # 计算z轴的偏角

    # compute measured tray poses
    tray_positions = np.zeros((n, 3))
    tray_orientations = np.zeros((n, 4))
    tray_angles = np.zeros(n)
    for i in range(n):
        p = tray_msgs[i].transform.translation
        tray_positions[i, :] = [p.x, p.y, p.z]
        Q = tray_msgs[i].transform.rotation
        orientation = np.array([Q.x, Q.y, Q.z, Q.w])
        tray_orientations[i, :] = orientation
        R = tanalysis.math.quat_to_rot(orientation)
        tray_angles[i] = np.arccos(z @ R @ z)

    # error between measured and modelled orientation
    orientation_errors = np.zeros((n, 4))
    angle_errors = np.zeros(n)
    for i in range(n):
        R1 = tanalysis.math.quat_to_rot(ee_orientations[i, :])
        R2 = tanalysis.math.quat_to_rot(tray_orientations[i, :])
        ΔQ = tanalysis.math.rot_to_quat(R1 @ R2.T)
        orientation_errors[i, :] = ΔQ
        angle_errors[i] = tanalysis.math.quat_angle(ΔQ)

    # EE (model) position vs. time
    plt.figure()
    plt.plot(ts, ee_positions[:, 0], label="x")
    plt.plot(ts, ee_positions[:, 1], label="y")
    plt.plot(ts, ee_positions[:, 2], label="z")
    plt.xlabel("Time (s)")
    plt.ylabel("EE position (m)")
    plt.title(f"EE position vs. time")
    plt.legend()
    plt.grid()

    # EE (model) quaternion orientation vs. time
    plt.figure()
    plt.plot(ts, ee_orientations[:, 0], label="x")
    plt.plot(ts, ee_orientations[:, 1], label="y")
    plt.plot(ts, ee_orientations[:, 2], label="z")
    plt.plot(ts, ee_orientations[:, 3], label="w")
    plt.plot(ts, ee_angles, label="angle")
    plt.xlabel("Time (s)")
    plt.ylabel("EE orientation")
    plt.title(f"EE orientation vs. time")
    plt.legend()
    plt.grid()

    # Tray (measured) position vs. time
    plt.figure()
    plt.plot(ts, tray_positions[:, 0], label="x")
    plt.plot(ts, tray_positions[:, 1], label="y")
    plt.plot(ts, tray_positions[:, 2], label="z")
    plt.xlabel("Time (s)")
    plt.ylabel("Tray position (m)")
    plt.title(f"Tray position vs. time")
    plt.legend()
    plt.grid()

    # Tray (measured) quaternion orientation vs. time
    plt.figure()
    plt.plot(ts, tray_orientations[:, 0], label="x")
    plt.plot(ts, tray_orientations[:, 1], label="y")
    plt.plot(ts, tray_orientations[:, 2], label="z")
    plt.plot(ts, tray_orientations[:, 3], label="w")
    plt.plot(ts, tray_angles, label="angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Tray orientation")
    plt.title(f"Tray orientation vs. time")
    plt.legend()
    plt.grid()

    # Orientation error between model and measured values
    plt.figure()
    plt.plot(ts, orientation_errors[:, 0], label="x")
    plt.plot(ts, orientation_errors[:, 1], label="y")
    plt.plot(ts, orientation_errors[:, 2], label="z")
    plt.plot(ts, orientation_errors[:, 3], label="w")
    plt.plot(ts, angle_errors, label="angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Orientation error")
    plt.title(f"Orientation error vs. time")
    plt.legend()
    plt.grid()

    # Position error between model and measured values
    position_errors = tray_positions - ee_positions
    plt.figure()
    plt.plot(ts, position_errors[:, 0], label="x")
    plt.plot(ts, position_errors[:, 1], label="y")
    plt.plot(ts, position_errors[:, 2], label="z")
    plt.xlabel("Time (s)")
    plt.ylabel("EE Position error (m)")
    plt.title(f"EE Position error between model and measured values")
    plt.legend()
    plt.grid()

    plt.show()

    plt.show()


if __name__ == "__main__":
    main()
