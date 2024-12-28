#!/usr/bin/env python3
"""Print positions of markers near a reference point."""
import rospy
import numpy as np

from vicon_bridge.msg import Markers



# positions to check [meters]
# POSITIONS = np.array([[0, 0.25, 1], [1.5, 1, 1], [-0.5, 2, 1]])
POSITIONS = np.array([[0, 0, 0]])
# 定义参考点的坐标列表，单位是米。代码中只定义了一个点 [0, 0, 0]。

# print marker position if it is within this distance [meters] of one of the
# above positions
RADIUS = 0.2
# 如果标记点与参考点的距离小于 0.2 米，则打印该标记点位置。

# Set entries to False to mask out x, y, or z components, respectively
COORDINATE_MASK = [True, True, False]
# 设置用于屏蔽坐标轴的掩码。当前设置忽略 Z 轴，仅考虑 X 和 Y 轴。

# Set True to only print out markers that do not belong to an object
ONLY_PRINT_UNATTACHED_MARKERS = True
# 如果设置为 True，则仅打印未附加到对象的标记点（marker_name 为空）。


class ViconMarkerPrinter:
    def __init__(self):
        """Print out markers near reference points."""
        self.marker_sub = rospy.Subscriber(
            "/vicon/markers", Markers, self._marker_cb)
        # 订阅 `/vicon/markers` 主题，回调函数为 `_marker_cb`。

    def _marker_cb(self, msg):
        positions = []
        for marker in msg.markers:
            if ONLY_PRINT_UNATTACHED_MARKERS and marker.marker_name != "":
                continue
            # 如果标记点有 marker_name 并且 `ONLY_PRINT_UNATTACHED_MARKERS` 为 True，则跳过该标记点。
            #! 作用就是找到哪些没有名字的marker球，是不是就是找到那些没有使用的(没有用来组成vicon对象的)marker球

            p = marker.translation
            p = np.array([p.x, p.y, p.z]) / 1000  # convert to meters
            # 获取标记点的位置（毫米），并转换为米。

            # get offset, mask out coordinates we don't care about
            r = (POSITIONS - p) * COORDINATE_MASK
            # 计算标记点与每个参考点的偏移量，并根据 COORDINATE_MASK 屏蔽不感兴趣的坐标分量。

            # distance from each point
            d = np.sum(r**2, axis=1)
            # 计算标记点与每个参考点的平方距离。

            if np.any(d <= RADIUS**2):
                positions.append(p)
                # 如果距离小于等于 RADIUS，记录标记点的位置。

        # sort by x-position
        if len(positions) > 0:
            positions = np.array(positions)
            idx = np.argsort(positions[:, 0])
            # 按 X 坐标升序排序。

            print(positions[idx, :])
            # 打印排序后的标记点位置。
        else:
            print("no markers found")
            # 如果没有符合条件的标记点，打印提示信息。


def main():
    np.set_printoptions(precision=6, suppress=True)
    # 设置打印选项，精度为 6 位小数，禁用科学计数法。

    rospy.init_node("vicon_marker_printer")
    # 初始化 ROS 节点，名称为 `vicon_marker_printer`。

    printer = ViconMarkerPrinter()
    # 创建 ViconMarkerPrinter 对象。

    rospy.spin()
    # 保持节点运行，等待消息。


if __name__ == "__main__":
    main()
# 如果脚本是直接运行的，则调用 main() 函数。
