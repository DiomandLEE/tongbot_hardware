#!/usr/bin/env python3
"""Plot end effector position, velocity, and acceleration from a bag file.

Also computes the maximum velocity and acceleration.
"""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import signal
import seaborn

from tongbot_hardware_central import ros_utils

import IPython

'''
读取bag中的vicon
这个代码的主要作用是使用 Savitzky-Golay 滤波器 对模拟的机械臂末端执行器（EE）的 位置、速度 和 加速度 数据进行平滑处理，
并且通过 matplotlib 提供的滑块控件来动态调整滤波器的参数（窗口大小和多项式阶数），实时观察数据平滑效果的变化。
后处理，存了一波位置数据
'''


VICON_OBJECT_NAME = "ThingWoodTray" # VICON运动捕捉系统的目标物名称，用于指定 ROS 中的目标。

# Savitzky-Golay 滤波函数
# 平滑效果：去除高频噪声，同时保留信号的趋势和局部特征。
# 可计算导数：S-G 滤波可以同时用来计算数据的一阶或更高阶导数，这是它的一个重要特性。
def savgol(x, y, window_length, polyorder, deriv=0):
    """Savgol filter for non-uniformly spaced data.

    Much slower than scipy.signal.savgol_filter because we need to refit the
    polynomial to every window of data.
    实现 Savitzky-Golay 滤波，用于不均匀间隔数据。相比 Scipy 的内置函数速度较慢，因为需要每次对窗口数据重新拟合多项式。
    """
    assert len(x) == len(y)
    assert window_length > polyorder
    assert type(deriv) == int and deriv >= 0
    # x 和 y 的长度必须相等。
    # 滤波窗口必须大于多项式阶数。
    # 求导次数 deriv 必须是非负整数

    r = window_length // 2
    n = len(x)
    degree = polyorder - deriv

    y_smooth = np.zeros_like(y)
    # r：窗口半径。
    # n：数据点总数。
    # degree：实际拟合多项式的阶数（多项式阶数减去求导次数）。
    # y_smooth：初始化输出数组，与 y 形状相同。
    for i in range(n):
        low = max(i - r, 0)
        high = min(i + r + 1, n)
        x_window = x[low:high]
        y_window = y[low:high]
        poly = np.polynomial.Polynomial.fit(x_window, y_window, polyorder)
        for _ in range(deriv):
            poly = poly.deriv()
        y_smooth[i] = poly(x[i])
    return y_smooth
    # 计算当前窗口范围。
    # 提取窗口内的数据。
    # 使用窗口数据拟合多项式。
    # 根据指定的 deriv 次数对多项式求导。
    # 计算当前点的平滑值。

# 使用 Matplotlib 滑块实时调整滤波参数的类。
class FilterUpdater:
    """Add sliders to a plot to control SavGol filter window size and poly order."""
    def __init__(
        self, fig, lines, positions, window_size, polyorder, delta=1.0, deriv=0
    ):
        # fig：要更新的图形。
        # lines：图形上的线条。
        # positions：原始位置数据。
        # window_size 和 polyorder：滤波参数。
        # delta：时间间隔。
        # deriv：导数阶数。
        self.fig = fig
        self.lines = lines
        self.positions = positions
        self.window_size = window_size
        self.polyorder = polyorder
        self.delta = delta
        self.deriv = deriv
        # 定义滑块的显示位置。
        ax_window_slider = fig.add_axes([0.35, 0.15, 0.5, 0.03])
        ax_poly_slider = fig.add_axes([0.35, 0.1, 0.5, 0.03])
        # 创建滑块对象。
        self.window_slider = Slider(
            ax_window_slider, "Window Size", 5, 100, valinit=window_size, valstep=1
        )
        self.poly_slider = Slider(
            ax_poly_slider, "Poly Order", 1, 10, valinit=polyorder, valstep=1
        )
        # 为滑块添加事件监听，滑块变化时调用对应更新函数。
        self.window_slider.on_changed(self.update_window_size)
        self.poly_slider.on_changed(self.update_polyorder)

    def update(self):
        y = signal.savgol_filter(
            self.positions,
            self.window_size,
            self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
            axis=0,
        ) # 使用 Scipy 的 Savitzky-Golay 滤波器对 positions 数据进行滤波。
        for i, line in enumerate(self.lines[:3]):
            line.set_ydata(y[:, i])
        self.lines[-1].set_ydata(np.linalg.norm(y, axis=1))
        self.fig.canvas.draw_idle()
        # 更新图中曲线的数据并重绘。

    def update_window_size(self, window_size):
        self.window_size = window_size
        self.update()

    def update_polyorder(self, polyorder):
        self.polyorder = polyorder
        self.update()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()
    #! python plot—bag.py /path/to/your/data.bag

    bag = rosbag.Bag(args.bagfile)
    # 解析命令行参数，读取bag文件

    topic = ros_utils.vicon_topic_name(VICON_OBJECT_NAME)
    msgs = [msg for _, msg, _ in bag.read_messages(topic)] # 它返回一个生成器，每个元素都是一个三元组，包含消息的时间戳、消息本身和消息的元数据（例如来源等）。
    ts, poses = ros_utils.parse_transform_stamped_msgs(msgs, normalize_time=True) # 解析生成器
    # 获取 VICON 话题数据

    n = len(ts)
    positions = poses[:, :3] # 只取位置信息
    dt = np.mean(ts[1:] - ts[:-1])  # average time step 计算时间间隔 dt，即相邻时间戳之间的平均差值。假设数据是均匀采样的，这个值可以用于后续的数值计算

    # smoothed velocities and accelerations using Savitzky-Golay filter
    window_size = 31
    polyorder = 2
    smooth_velocities = signal.savgol_filter(
        positions, window_size, polyorder, deriv=1, delta=dt, axis=0
    )
    smooth_accelerations = signal.savgol_filter(
        positions, window_size, polyorder, deriv=2, delta=dt, axis=0
    )
    # 使用 Savitzky-Golay 滤波器对位置数据进行平滑处理。Savitzky-Golay 滤波器能够通过局部多项式拟合来平滑数据，并计算导数。

    # (noisy) velocities and accelerations computed by finite differences
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)
    for i in range(1, n - 1):
        velocities[i, :] = (positions[i + 1, :] - positions[i - 1, :]) / (
            ts[i + 1] - ts[i - 1]
        )

    for i in range(2, n - 2):
        accelerations[i, :] = (velocities[i + 1, :] - velocities[i - 1, :]) / (
            ts[i + 1] - ts[i - 1]
        )
    # 使用有限差分法计算位置数据的速度和加速度。有限差分法是一种数值计算方法，用于估计导数。这里是中心差分法，即使用相邻数据点的差值来估计导数。

    palette = seaborn.color_palette("deep") # 这行代码加载 seaborn 调色板，以便后续绘制的曲线使用一致的颜色。

    plt.figure()
    plt.plot(ts, positions[:, 0], label="x")
    plt.plot(ts, positions[:, 1], label="y")
    plt.plot(ts, positions[:, 2], label="z")
    plt.title("EE position")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.legend()
    plt.grid()

    vel_fig = plt.figure()
    plt.plot(ts, velocities[:, 0], color=palette[0], alpha=0.2)
    plt.plot(ts, velocities[:, 1], color=palette[1], alpha=0.2)
    plt.plot(ts, velocities[:, 2], color=palette[2], alpha=0.2)
    plt.plot(ts, np.linalg.norm(velocities, axis=1), color=palette[3], alpha=0.2)
    (l1,) = plt.plot(ts, smooth_velocities[:, 0], label="x", color=palette[0])
    (l2,) = plt.plot(ts, smooth_velocities[:, 1], label="y", color=palette[1])
    (l3,) = plt.plot(ts, smooth_velocities[:, 2], label="z", color=palette[2])
    (l4,) = plt.plot(ts, np.linalg.norm(smooth_velocities, axis=1), label="norm", color=palette[3])
    plt.title("EE velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend()
    plt.grid()

    vel_updater = FilterUpdater(
        vel_fig, [l1, l2, l3, l4], positions, window_size, polyorder, delta=dt, deriv=1
    )

    acc_fig = plt.figure()
    plt.plot(ts, accelerations[:, 0], color=palette[0], alpha=0.2)
    plt.plot(ts, accelerations[:, 1], color=palette[1], alpha=0.2)
    plt.plot(ts, accelerations[:, 2], color=palette[2], alpha=0.2)
    plt.plot(ts, np.linalg.norm(accelerations, axis=1), color=palette[3], alpha=0.2)
    (l1,) = plt.plot(ts, smooth_accelerations[:, 0], label="x", color=palette[0])
    (l2,) = plt.plot(ts, smooth_accelerations[:, 1], label="y", color=palette[1])
    (l3,) = plt.plot(ts, smooth_accelerations[:, 2], label="z", color=palette[2])
    (l4,) = plt.plot(ts, np.linalg.norm(smooth_accelerations, axis=1), label="norm", color=palette[3])
    plt.title("EE acceleration")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s]")
    plt.legend()
    plt.grid()

    acc_updater = FilterUpdater(
        acc_fig, [l1, l2, l3, l4], positions, window_size, polyorder, delta=dt, deriv=2
    )

    plt.show()


if __name__ == "__main__":
    main()
