#!/usr/bin/env python3
"""Continually publish zero velocity to stop the robot."""
import rospy

import tongbot_hardware_central as tongbot

RATE = 125  # Hz

def main():
    rospy.init_node("stop_node")
    robot = tongbot.MobileManipulatorROSInterface()
    rate = rospy.Rate(RATE)

    print("Publishing zero velocity commands until Ctrl-C is pressed...")
    while not rospy.is_shutdown():
        robot.brake()
        rate.sleep()


if __name__ == "__main__":
    main()
