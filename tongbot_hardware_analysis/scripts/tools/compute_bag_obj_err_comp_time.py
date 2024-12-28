#!/usr/bin/env python3
"""Plot robot true and estimated joint state from a bag file."""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt

from tongbot_hardware_central import ros_utils
from tongbot_hardware_analysis.parsing import parse_object_error, parse_mpc_solve_times

import IPython

'''
功能1: 从ROS的bag文件中提取MPC解算时间和误差数据。
功能2: 绘制上述数据的图表，以便分析和调试。

第一张图：绘制MPC解算时间随时间的变化曲线。
第二张图：绘制Vicon测量的误差（以毫米为单位）。
'''
#done 需要看upright_ros_interface.parsing中的parse_object_error, parse_mpc_solve_times


TRAY_VICON_NAME = "ThingWoodTray"


def get_bag_topics(bag):
    return list(bag.get_type_and_topic_info()[1].keys())


def vicon_object_topics(bag):
    topics = get_bag_topics(bag)

    def func(topic):
        if not topic.startswith("/vicon"):
            return False
        if (
            topic.endswith("markers")
            or topic.endswith("ThingBase")
            or topic.endswith(TRAY_VICON_NAME)
        ):
            return False
        return True

    topics = list(filter(func, topics))
    if len(topics) == 0:
        print("No object topic found!")
    elif len(topics) > 1:
        print("Multiple object topics found!")
    return topics[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)
    solve_times, ts1 = parse_mpc_solve_times(bag, max_time=5, return_times=True)
    object_vicon_name = vicon_object_topics(bag).split("/")[-1]
    print(f"Object is {object_vicon_name}")
    errors, ts2 = parse_object_error(
        bag, TRAY_VICON_NAME, object_vicon_name, return_times=True
    )

    print("SOLVE TIME")
    print(f"max  = {np.max(solve_times):.2f} ms")
    print(f"min  = {np.min(solve_times):.2f} ms")
    print(f"mean = {np.mean(solve_times):.2f} ms")

    plt.figure()
    plt.plot(ts1, solve_times)
    plt.xlabel("Time [s]")
    plt.ylabel("Solve time [ms]")
    plt.grid()

    plt.figure()
    plt.plot(ts2, 1000 * errors)  # convert to mm
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Distance error [mm]")

    plt.show()


if __name__ == "__main__":
    main()
