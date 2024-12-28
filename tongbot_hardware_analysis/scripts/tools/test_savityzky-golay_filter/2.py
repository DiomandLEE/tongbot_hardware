import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.signal import savgol_filter
from matplotlib.widgets import Slider
import time

'''
Savitzky-Golay 滤波器测试，实时处理，不是像1那样一次性后处理
'''

class FilterUpdater:
    def __init__(self, window_size=51, polyorder=3, noise_amplitude=0.05):
        """
        初始化滤波器更新器。

        参数：
        window_size: 滤波器窗口大小
        polyorder: 多项式阶数
        noise_amplitude: 噪声的幅度
        """
        self.window_size = window_size
        self.polyorder = polyorder
        self.noise_amplitude = noise_amplitude

    def true_system(self, t):
        """
        生成加速度、速度和位置数据，并添加噪声。

        参数：
        t: 时间序列

        返回：
        position, velocity, acceleration: 位置、速度和加速度
        """
        dt = t[1] - t[0] if len(t) > 1 else 0.01  # 使用时间序列的步长

        # 加权叠加多个三角函数来生成加速度
        freq1, amp1, phase1 = 0.5, 1.0, 0  # 第一个三角函数
        freq2, amp2, phase2 = 1.0, 0.5, np.pi / 4  # 第二个三角函数
        freq3, amp3, phase3 = 2.0, 0.2, np.pi / 2  # 第三个三角函数

        # 生成加速度（多个三角函数的叠加）
        acceleration = 0.3 * (
            amp1 * np.sin(freq1 * t + phase1) +
            amp2 * np.cos(freq2 * t + phase2) +
            amp3 * np.sin(freq3 * t + phase3)
        )

        # 计算速度和位置（累积积分），并加入噪声
        velocity = integrate.cumtrapz(acceleration, dx=dt, initial=0) + 0.5 * self.noise_amplitude * np.random.normal(0, 1, size=t.shape)
        position = integrate.cumtrapz(velocity, dx=dt, initial=0) + self.noise_amplitude * np.random.normal(0, 1, size=t.shape)

        return position, velocity, acceleration

    def update(self, t, noise_amplitude):
        """
        更新噪声幅度，并返回滤波后的数据。

        参数：
        t: 时间序列
        noise_amplitude: 当前噪声幅度

        返回：
        filtered_position, filtered_velocity, filtered_acceleration: 滤波后的数据
        """
        # 更新噪声幅度
        self.noise_amplitude = noise_amplitude

        # 获取真实系统的数据：位置、速度、加速度（带噪声）
        position, velocity, acceleration = self.true_system(t)

        # 处理窗口大小，确保不会超过数据的长度
        actual_window_size = min(len(t), self.window_size)

        # 确保窗口大小大于多项式阶数
        if actual_window_size <= self.polyorder:
            actual_window_size = self.polyorder + 1

        # 使用 Savitzky-Golay 滤波器进行平滑处理
        filtered_position = savgol_filter(position, actual_window_size, self.polyorder)
        filtered_velocity = savgol_filter(velocity, actual_window_size, self.polyorder, deriv=1)
        filtered_acceleration = savgol_filter(acceleration, actual_window_size, self.polyorder, deriv=2)

        return position, velocity, acceleration, filtered_position, filtered_velocity, filtered_acceleration


def plot_comparison(t, position, velocity, acceleration, filtered_position, filtered_velocity, filtered_acceleration):
    """
    绘制位置、速度和加速度的对比图。

    参数：
    t: 时间序列
    position, velocity, acceleration: 原始数据
    filtered_position, filtered_velocity, filtered_acceleration: 滤波后的数据
    """
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    # 绘制位置对比图
    ax[0].plot(t, position, label="Noisy Position", color='blue')
    ax[0].plot(t, filtered_position, label="Filtered Position", color='orange')
    ax[0].set_title("Position Comparison")
    ax[0].legend()

    # 绘制速度对比图
    ax[1].plot(t, velocity, label="Noisy Velocity", color='green')
    ax[1].plot(t, filtered_velocity, label="Filtered Velocity", color='red')
    ax[1].set_title("Velocity Comparison")
    ax[1].legend()

    # 绘制加速度对比图
    ax[2].plot(t, acceleration, label="True Acceleration", color='black')
    ax[2].plot(t, filtered_acceleration, label="Filtered Acceleration", color='purple')
    ax[2].set_title("Acceleration Comparison")
    ax[2].legend()

    plt.tight_layout()
    plt.show()


def update(val):
    """
    更新噪声幅度并重新绘制对比图。

    参数：
    val: 当前滑块的值
    """
    noise_amplitude = slider.val

    # 获取滤波器和系统数据
    position, velocity, acceleration, filtered_position, filtered_velocity, filtered_acceleration = filter_updater.update(t, noise_amplitude)

    # 绘制对比图
    plot_comparison(t, position, velocity, acceleration, filtered_position, filtered_velocity, filtered_acceleration)


def main():
    """
    主函数，生成数据并进行滤波和对比绘制。
    """
    # 初始化时间序列
    global t
    t = np.linspace(0, 10, 1000)

    # 创建 FilterUpdater 实例
    global filter_updater
    filter_updater = FilterUpdater(window_size=51, polyorder=3, noise_amplitude=0.05)

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 创建滑块
    ax_slider = plt.axes([0.1, 0.02, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    global slider
    slider = Slider(ax_slider, 'Noise Amplitude', 0.01, 0.1, valinit=0.05, valstep=0.01)

    # 添加滑块更新事件
    slider.on_changed(update)

    # 初始化图形并开始实时更新
    t_new = np.linspace(0, 10, 1000)  # 确保时间序列与数据长度匹配
    for i in range(1000):
        time.sleep(0.01)  # 模拟实时获取数据的延迟

        # 获取滤波器和系统数据
        position, velocity, acceleration, filtered_position, filtered_velocity, filtered_acceleration = filter_updater.update(t_new, slider.val)

        # 绘制对比图
        plot_comparison(t_new, position, velocity, acceleration, filtered_position, filtered_velocity, filtered_acceleration)

    plt.show()


if __name__ == "__main__":
    main()
