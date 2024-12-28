"参考plot_joint_states.py进行修改"
"""Plot kinova and dingo joint position and velocity from a ROS bag."""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from tongbot_hardware_central import ros_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)

    kinova_msgs = [msg for _, msg, _ in bag.read_messages("/my_gen3/joint_states")]
    dingo_msgs = [msg for _, msg, _ in bag.read_messages("/dingo/joint_states")]

    kinova_cmd_msgs = [msg for _, msg, _ in bag.read_messages("/my_gen3/cmd_vel")]
    kinova_cmd_ts = np.array([t.to_sec() for _, _, t in bag.read_messages("/my_gen3/cmd_vel")])
    kinova_cmd_ts -= kinova_cmd_ts[0]

    # TODO trim messages to only start once we get a command
    kinova_cmd_vels = []
    for msg in kinova_cmd_msgs:
        kinova_cmd_vels.append(msg.data)
    kinova_cmd_vels = np.array(kinova_cmd_vels)

    tas, qas, vas = ros_utils.parse_kinova_joint_state_msgs(kinova_msgs)
    tbs, qbs, vbs = ros_utils.parse_dingo_joint_state_msgs(dingo_msgs)

    plt.figure()
    plt.plot(tbs, qbs[:, 0], label="x")
    plt.plot(tbs, qbs[:, 1], label="y")
    plt.plot(tbs, qbs[:, 2], label="θ")
    plt.title("Dingo Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(tbs, vbs[:, 0], label="x")
    plt.plot(tbs, vbs[:, 1], label="y")
    plt.plot(tbs, vbs[:, 2], label="θ")
    plt.title("Dingo Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(7):
        plt.plot(tas, qas[:, i], label=f"θ_{i+1}")
    plt.title("Kinova Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position (rad)")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(7):
        plt.plot(tas, vas[:, i], label=f"v_{i+1}")
    plt.title("Kinova Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity (rad/s)")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(7):
        plt.plot(kinova_cmd_ts, kinova_cmd_vels[:, i], label=f"vc_{i+1}")
    plt.title("Kinova Commanded Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity (rad/s)")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
