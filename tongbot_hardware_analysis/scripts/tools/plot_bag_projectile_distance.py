#!/usr/bin/env python3
"""Plot distance of projectile to:
1. initial position of the EE (to verify that it would have hit the EE)
2. actual position of the EE (to verify that the EE actually got out of the way)

Also computes the maximum velocity and acceleration.
"""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt

from tongbot_hardware_central import ros_utils

import IPython

'''
通过插值对齐不同数据流的时间戳。
利用距离计算分析投射物的运动和 EE 是否有效躲避。
以可视化方式验证投射物的行为和 EE 的响应。
'''
#todo 看ros_utils的parse_transform_stamped_msgs、vicon_topic_name函数
#todo 可以再加上很多个距离，比如各个joint关节，已经底盘，这个就直接在kinova上贴marker就行


# TODO we would prefer to measure offset to the actual collision sphere
#   我们更倾向于测量到实际碰撞球体的偏移量
VICON_PROJECTILE_NAME = "Dynamic_Obs"
VICON_EE_NAME = "gripper_robotiq2"
#todo 使用的时候需要修改这些，还有topic名称


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)
    # 使用终端命令行指定 ROS bag 文件作为输入。

    ee_topic = ros_utils.vicon_topic_name(VICON_EE_NAME)
    ee_msgs = [msg for _, msg, _ in bag.read_messages(ee_topic)]
    ee_ts, ee_poses = ros_utils.parse_transform_stamped_msgs(
        ee_msgs, normalize_time=False
    )
    # 根据指定的 Vicon 名称提取 EE 的位置信息，并解析为时间戳（ee_ts）和位姿数据（ee_poses）。

    projectile_topic = ros_utils.vicon_topic_name(VICON_PROJECTILE_NAME)
    projectile_msgs = [msg for _, msg, _ in bag.read_messages(projectile_topic)]
    projectile_ts, projectile_poses = ros_utils.parse_transform_stamped_msgs(
        projectile_msgs, normalize_time=False
    )
    # 类似于 EE，提取投射物的时间戳和位姿数据。但是这个位置是直接从vicon的topic中获取的。

    proj_est_msgs = [
        msg for _, msg, _ in bag.read_messages("/projectile/joint_states")
    ]
    proj_est_ts = ros_utils.parse_time(proj_est_msgs, normalize_time=False)
    proj_pos_est = np.array([msg.position for msg in proj_est_msgs])
    # 类似于 EE，提取投射物的时间戳和位姿数据。但是这个位置是直接从卡尔曼滤波后的结果，发布出来的topic中获取的。

    ts = ee_ts
    n = len(ts)
    projectile_positions = np.array(
        ros_utils.interpolate_list(ts, projectile_ts, projectile_poses[:, :3])
    )
    ee_positions = ee_poses[:, :3]
    proj_pos_est = np.array(ros_utils.interpolate_list(ts, proj_est_ts, proj_pos_est))
    ts -= ts[0]
    # 使用插值将投射物和末端执行器的数据对齐到统一的时间轴 ts。
    # ts 是以 EE 时间戳为基准，并调整为从 0 开始。

    # TODO may be easier to compute the EE position using the joint states
    # (possibly the estimate)---then we can directly get the collision sphere
    # position using the model
    # 可能通过使用关节状态（可能是估计值）来计算末端执行器(EE)的位置会更简单 ---然后我们可以直接使用模型得到碰撞球体的位置

    # only start when robot sees the ball when z >= 0
    # z大于0时，才视为看见投射物，如果我们要用的话，其实也没什么影响
    start_idx = np.argmax(proj_pos_est[:, 2] >= 0.0)

    r0 = ee_positions[0, :]
    distance_to_origin = np.linalg.norm(projectile_positions - r0, axis=1)
    distance_to_ee = np.linalg.norm(projectile_positions - ee_positions, axis=1)
    distance_to_origin_est = np.linalg.norm(proj_pos_est - r0, axis=1)
    distance_to_ee_est = np.linalg.norm(proj_pos_est - ee_positions, axis=1)
    # 计算以下四种距离：
    #     投射物到初始 EE 位置（r0）的实际距离和估计距离。
    #     投射物到实时 EE 位置的实际距离和估计距离。

    plt.figure()
    plt.plot(ts[start_idx:], distance_to_origin[start_idx:], label="Dist to Origin")
    plt.plot(ts[start_idx:], distance_to_ee[start_idx:], label="Dist to EE")
    plt.plot(
        ts[start_idx:],
        distance_to_origin_est[start_idx:],
        label="Est. Dist to Origin",
        linestyle="--",
    )
    plt.plot(
        ts[start_idx:],
        distance_to_ee_est[start_idx:],
        label="Est. Dist to EE",
        linestyle="--",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.title("Projectile Distance")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
