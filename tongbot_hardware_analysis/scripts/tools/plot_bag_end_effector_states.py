#!/usr/bin/env python3
"""Plot end effector position, velocity, and acceleration from a bag file.

Also computes the maximum velocity and acceleration.
"""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt

import tongbot_hardware_analysis as tanalysis
from tongbot_hardware_central import MobileManipulatorKinematics

'''
从ROS bag文件中提取机器人末端执行器的运动数据，包括位置、速度和加速度，并将其绘制出来。
它还计算了末端执行器的最大速度和最大加速度，并在控制台打印出这些最大值和对应的时间点。
'''
#todo 这个要是需要的话，有一大部分是需要自己重新修改的

def main():
    parser = argparse.ArgumentParser()
    tanalysis.cli.add_bag_dir_arguments(parser)
    config_path, bag_path = tanalysis.cli.parse_bag_dir_args(parser.parse_args())

    # config, model = tanalysis.parsing.parse_config_and_control_model(config_path)
    # robot = model.robot # 这里可以换成centrl里的kinematics，这个可以看一下src/upright_control/src/upright_control/robot.py也差不多
    robot = MobileManipulatorKinematics() #read 是要在这里面，修改robot urdf的文件位置

    bag = rosbag.Bag(bag_path)

    mpc_obs_msgs = [
        msg for _, msg, _ in bag.read_messages("/mobile_manipulator_mpc_observation") #todo
    ]
    ts, xs, us = tanalysis.parsing.parse_mpc_observation_msgs(mpc_obs_msgs, normalize_time=True)
    #这里是state，不是pinocchio position

    n = len(ts)

    ee_poses = np.zeros((n, 7))
    ee_velocities = np.zeros((n, 6))
    ee_accelerations = np.zeros((n, 6))

    for i in range(n):
        # 判断 xs[i,:] 的长度来决定如何提取数据
        if len(xs[i, :]) == 10:
        # 存储前 10 个数据
            q = xs[i, :10]
            robot.forward(q)
            # print("前10个数据:", first_10)
        elif len(xs[i, :]) == 20:
            # 存储前 10 个数据和后 10 个数据
            q = xs[i, :10]
            v = xs[i, -10:]
            robot.forward(q, v)
            # print("前10个数据:", first_10)
            # print("后10个数据:", last_10)
        elif len(xs[i, :]) == 30:
            # 存储前 10 个数据，中间 10 个数据和后 10 个数据
            q = xs[i, :10]
            v = xs[i, len(xs[i, :])//2 - 5 : len(xs[i, :])//2 + 5]
            a = xs[i, -10:]
            robot.forward(q, v, a)
        #     print("前10个数据:", first_10)
        #     print("中间10个数据:", middle_10)
        #     print("后10个数据:", last_10)
        else:
            print("无法处理该行数据，长度为", len(xs[i, :]))

        robot.forward_xu(xs[i, :])
        ee_poses[i, :] = np.concatenate(robot.link_pose())
        ee_velocities[i, :] = np.concatenate(robot.link_velocity())
        ee_accelerations[i, :] = np.concatenate(robot.link_classical_acceleration())

    # reference position is relative to the initial position
    # ref = config["controller"]["waypoints"][0]["position"] + ee_poses[0, :3]
    ref = ee_poses[0, :3] #目前没什么作用

    velocity_magnitudes = np.linalg.norm(ee_velocities[:, :3], axis=1)
    max_vel_idx = np.argmax(velocity_magnitudes)
    max_vel = velocity_magnitudes[max_vel_idx]
    print(f"Max velocity = {max_vel:.3f} m/s at time = {ts[max_vel_idx]} seconds.")

    acceleration_magnitudes = np.linalg.norm(ee_accelerations[:, :3], axis=1)
    max_acc_idx = np.argmax(acceleration_magnitudes)
    max_acc = acceleration_magnitudes[max_acc_idx]
    print(
        f"Max acceleration = {max_acc:.3f} m/s^2 at time = {ts[max_acc_idx]} seconds."
    )

    plt.figure()
    lx, = plt.plot(ts, ee_poses[:, 0], label="x")
    ly, = plt.plot(ts, ee_poses[:, 1], label="y")
    lz, = plt.plot(ts, ee_poses[:, 2], label="z")
    plt.axhline(ref[0], label="xd", linestyle="--", color=lx.get_color())
    plt.axhline(ref[1], label="yd", linestyle="--", color=ly.get_color())
    plt.axhline(ref[2], label="zd", linestyle="--", color=lz.get_color())
    plt.title("EE position")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(ts, ee_velocities[:, 0], label="x")
    plt.plot(ts, ee_velocities[:, 1], label="y")
    plt.plot(ts, ee_velocities[:, 2], label="z")
    plt.title("EE velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(ts, ee_accelerations[:, 0], label="x")
    plt.plot(ts, ee_accelerations[:, 1], label="y")
    plt.plot(ts, ee_accelerations[:, 2], label="z")
    plt.title("EE acceleration")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s^2]")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
