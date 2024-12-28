#!/usr/bin/env python3
"""Print location of Vicon markers near particular locations.

This is useful for e.g. ensuring the location of obstacles is correct in the
real world.
"""
import numpy as np
import rospy
from vicon_bridge.msg import Markers


# positions to check
POSITIONS = np.array([[0, 0.25, 1], [1.5, 1, 1], [-0.5, 2, 1]])
# POSITIONS 是一个 Numpy 数组，包含了三个目标位置，每个位置都是一个三维坐标，表示空间中的特定点。

# print marker position if it is within this distance of one of the above
# positions (one per row)
# 如果标记的位置在上述位置中的某一个的这个距离范围内，则打印该标记的位置（每个位置占一行）
RADIUS = 0.5
# RADIUS 定义了一个半径，用于判断 Vicon 标记与目标位置之间的距离。如果标记距离任何目标位置小于该半径（0.5 米），就认为它接近该位置。

class ViconMarkerPrinter:
    def __init__(self):
        self.marker_sub = rospy.Subscriber("/vicon/markers", Markers, self._marker_cb)

    def _marker_cb(self, msg):
        for marker in msg.markers:
            r = marker.translation
            r = np.array([r.x, r.y, r.z]) / 1000  # convert to meters
            if (np.linalg.norm(POSITIONS - r, axis=1) < RADIUS).any():
                print(f"Marker {marker.marker_name} at position = {r}")


def main():
    rospy.init_node("vicon_marker_printer")
    printer = ViconMarkerPrinter()
    rospy.spin()


main()

'''
. 机器人导航与障碍物检测
    应用场景：在多机器人协作任务中，多个机器人需要在同一环境中导航。
    如果环境中存在障碍物或目标物体（例如机器人的工作区域、路径或目标位置），你需要确保这些物体的位置在 Vicon 系统中被准确地捕捉。
示例：
    假设你有一个机器人，它需要避开特定的障碍物并到达指定的目标位置。你可以使用这个脚本来验证障碍物的位置是否与预定位置一致。
    例如，目标位置 POSITIONS = [[0, 0.25, 1], [1.5, 1, 1], [-0.5, 2, 1]] 代表了多个障碍物的位置，
    你希望 Vicon 系统能够准确捕捉这些物体的位置，确保机器人在进行路径规划时可以正确避开障碍物。

    如果你仅仅是为了大致定位障碍物或目标物体的位置，并且不需要非常精确的测量，0.5 米可能是合理的
'''
