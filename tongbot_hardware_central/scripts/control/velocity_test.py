#!/usr/bin/env python3
"""Send a constant velocity to a single joint for a given duration to test the system response."""
import argparse
import numpy as np
import rospy

'''
你想给第 3 个关节发送速度指令，并且不进行实际操作而只是打印命令，你可以在终端中输入：

bash
复制代码
rosrun mobile_manipulation velocity_test.py 3 --dry-run
如果你希望实际发送命令并控制机器人，去掉 --dry-run 参数：

bash
复制代码
rosrun mobile_manipulation velocity_test.py 3
确保你已经启动了 ROS 节点，并且机器人系统已经连接好。
'''

import tongbot_hardware_central as tongbot

VELOCITY = 0.2
DURATION = 2.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "joint_index",
        help="Index of the robot joint to send velocity step to.",
        type=int,
    )
    parser.add_argument(
        "--dry-run",
        help="Don't send any commands, just print out what would be sent.",
        action="store_true",
    )
    args = parser.parse_args()

    rospy.init_node("velocity_test")

    robot = tongbot.MobileManipulatorROSInterface()

    # wait until robot feedback has been received
    rate = rospy.Rate(125)
    while not rospy.is_shutdown() and not robot.ready():
        rate.sleep()

    cmd_vel = np.zeros(robot.nv)
    cmd_vel[args.joint_index] = VELOCITY

    # send command, wait, then brake
    if args.dry_run:
        print(cmd_vel)
    else:
        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            t = rospy.Time.now().to_sec()
            if t - t0 >= DURATION:
                break
            robot.publish_cmd_vel(cmd_vel)
            rate.sleep()
    robot.brake()


if __name__ == "__main__":
    main()
