#!/usr/bin/env python3
"""Plot robot true and estimated joint state from a bag file."""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt

from tongbot_hardware_central import ros_utils
from tongbot_hardware_analysis.parsing import parse_mpc_observation_msgs

#done 需要看upright_ros_interface.parsing中的parse_mpc_observation_msgs函数

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()
    #! 通过 argparse 解析命令行参数，获取 bag 文件路径。

    bag = rosbag.Bag(args.bagfile)

    kinova_msgs = [msg for _, msg, _ in bag.read_messages("/my_gen3/joint_states")]
    dingo_msgs = [msg for _, msg, _ in bag.read_messages("/dingo/joint_states")]
    mpc_obs_msgs = [
        msg for _, msg, _ in bag.read_messages("/mobile_manipulator_mpc_observation")
    ]
    mpc_plan_msgs = [
        msg for _, msg, _ in bag.read_messages("/mobile_manipulator_mpc_plan")
    ] #todo 跟着tongbot_mpc_node去修改这个topic名称
    '''
    使用 rosbag.Bag 读取 ROS bag 文件中的消息。
    分别读取 kinova/joint_states（UR10 机器人的关节状态）、dingo/joint_states（Ridgeback 基座的关节状态）、
    mobile_manipulator_mpc_observation（MPC 观测数据）和 mobile_manipulator_mpc_plan（MPC plan结果数据）
    '''

    tas, qas, vas = ros_utils.parse_kinova_joint_state_msgs(
        kinova_msgs, normalize_time=False
    )
    #! 从后面的处理来看，都是用arm的时间来对齐的，所以感觉是arm的joint_state是最先发出来的。
    tbs, qbs, vbs = ros_utils.parse_dingo_joint_state_msgs(
        dingo_msgs, normalize_time=False
    )
    #! 通过 ros_utils 模块的 parse_kinova_joint_state_msgs 和 parse_dingo_joint_state_msgs 函数
    #! 解析机器人关节的时间、位置和速度数据。

    # alternatively we can use finite differences, since the vbs are already
    # low-pass filtered otherwise
    # vbs = np.zeros_like(qbs)
    # for i in range(1, vbs.shape[0]):
    #     vbs[i, :] = (qbs[i, :] - qbs[i - 1, :]) / (tbs[i] - tbs[i - 1])
    '''
    # 或者我们可以使用有限差分法，因为 vbs 已经是低通滤波过的
    # vbs = np.zeros_like(qbs)
    # 对于每一行（从第二行开始），计算 vbs 的值
    #     vbs[i, :] = (qbs[i, :] - qbs[i - 1, :]) / (tbs[i] - tbs[i - 1])
    。常见的低通滤波方法包括：
        简单移动平均滤波器
        指数加权移动平均滤波器
        数字滤波器（如Butterworth、Chebyshev滤波器）
    '''

    tms, xms, ums = parse_mpc_observation_msgs(mpc_obs_msgs, normalize_time=False) # measured
    tps, xps, ups = parse_mpc_observation_msgs(mpc_plan_msgs, normalize_time=False) # planned
    #! 使用 parse_mpc_observation_msgs 函数解析 MPC 观测数据和计划数据，分别获得时间、状态和控制输入。

    # use arm messages for timing
    # TODO no need to include anything before the first policy message is
    # received
    # 使用臂部消息来进行时间同步
    # TODO 在接收到第一个策略消息之前，不需要包括任何内容
    # ts = tas
    # 只有在第一个观测值发布之前才开始

    ts = tas #arm的时间

    # only start just before the first observation is published
    start_idx = np.argmax(ts >= tms[0]) - 1 # 这行代码的作用是找到 ts 数组中第一个时间戳大于或等于 tms[0] 的位置，并通过 -1 来确定正确的起始索引。
    # ts = [0.5, 1.0, 1.5, 2.0, 2.5] （UR10 机器人状态的时间戳）
    # tms = [1.2, 1.8, 2.3] （MPC 观测数据的时间戳） 比如，这个，若是直接取np.argmax(ts >= tms[0])，那么就超过了tms的初始位置
    assert start_idx >= 0

    ts = ts[start_idx:]
    qas = qas[start_idx:, :]
    vas = vas[start_idx:, :]
    #! 通过查找 MPC 观测数据的第一个时间戳，截取并对齐机器人关节的状态数据（UR10 和 Ridgeback 的数据）。

    # import IPython
    # IPython.embed()
    # return

    # align base messages with the arm messages
    qbs_aligned = ros_utils.interpolate_list(ts, tbs, qbs)
    vbs_aligned = ros_utils.interpolate_list(ts, tbs, vbs)
    # 使用 ros_utils.interpolate_list 函数根据时间戳对base的数据进行插值对齐。

    qs_real = np.hstack((qbs_aligned, qas))
    vs_real = np.hstack((vbs_aligned, vas))
    # 将base和arm的q和v，直接拼接在一起，作为真实q和v

    # qs_real = qas
    # vs_real = vas

    # align the estimates and input
    n = 10
    xms_aligned = np.array(ros_utils.interpolate_list(ts, tms, xms))
    ums_aligned = np.array(ros_utils.interpolate_list(ts, tms, ums))
    # 对齐测量的mpc state： 对 MPC 的观测数据进行插值，提取关节位置、速度、加速度等信息，生成 qs_obs、vs_obs 和 as_obs。
    qs_obs = xms_aligned[:, :n]
    vs_obs = xms_aligned[:, n:2*n]
    as_obs = xms_aligned[:, 2*n:3*n] #感觉并没有考虑障碍物，可能是observation没管它
    us_obs = ums_aligned[:, :n]

    # align MPC optimal trajectory
    xps_aligned = np.array(ros_utils.interpolate_list(ts, tps, xps))
    # 对齐mpc plan出的state：  同样，对 MPC plan的数据进行插值，生成对应的关节位置、速度和加速度。
    qs_plan = xps_aligned[:, :n]
    vs_plan = xps_aligned[:, n:2*n]
    as_plan = xps_aligned[:, 2*n:3*n]

    ts -= ts[0] # 转化为相对时间，从0开始

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    plt.figure()
    for i in range(n):
        plt.plot(ts, qs_real[:, i], label=f"$q_{i+1}$")
    plt.title("Real Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, qs_obs[:, i], label=f"$\hat{{q}}_{i+1}$")
    for i in range(n):
        plt.plot(ts, qs_plan[:, i], label=f"$q^{{plan}}_{i+1}$", linestyle="--", color=colors[i])
    plt.title("Estimated Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend(ncols=2)
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, qs_real[:, i] - qs_obs[:, i], label=f"$q_{i+1}$")
    plt.title("Joint Position Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs_real[:, i], label=f"$v_{i+1}$")
    plt.title("Real Joint Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs_obs[:, i], label=f"$\hat{{v}}_{i+1}$")
    for i in range(n):
        plt.plot(ts, vs_plan[:, i], label=f"$v^{{plan}}_{i+1}$", linestyle="--", color=colors[i])
    plt.title("Estimated and Planned Joint Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend(ncol=2)
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs_real[:, i] - vs_obs[:, i], label=f"$v_{i+1}$")
    plt.title("Joint Velocity Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, as_obs[:, i], label=f"$\hat{{a}}_{i+1}$")
    for i in range(n):
        plt.plot(ts, as_plan[:, i], label=f"$a^{{plan}}_{i+1}$", linestyle="--", color=colors[i])
    plt.title("Estimated Joint Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint acceleration")
    plt.legend(ncol=2)
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, us_obs[:, i], label=f"$j_{i+1}$")
    plt.title("Joint (Jerk) Input")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint jerk")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
