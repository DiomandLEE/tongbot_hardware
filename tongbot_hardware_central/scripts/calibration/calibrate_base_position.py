#!/usr/bin/env python3
"""
使用 `collect_base_calibration_data.py` 收集的数据校准基座。

该脚本计算一个偏移量 `r_bv_v`，以修正测得的 Vicon 姿态，确保基座坐标系原点
在纯旋转时尽可能保持静止。
方法：
找到marker距离base中点的x和y方向的距离，让base旋转，
如果marker和base中心重合的话，那么maker点的x和y方向是不会有变化的
"""

import argparse  # 用于解析命令行参数
from spatialmath.base import rotz, q2r  # 提供旋转矩阵和四元数转换功能
import numpy as np  # 用于数值计算

# Vicon 的零位姿，用四元数表示，稍后用于旋转到正确的基座坐标系中
Q_bv = np.array([0, 0, -0.00792683, 0.99996858])
#! 即bQv，表示vicon坐标系相对于base坐标系的旋转，用于vicon-base系的坐标变换到real-base系


def main():
    # 解析命令行参数，接收输入文件
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="包含校准数据的 npz 文件。")
    args = parser.parse_args()

    # 加载校准数据文件
    data = np.load(args.filename)  # 从 npz 文件中加载数据
    q0 = data["q0"]  # 初始姿态
    qs = data["qs"]  # 所有测量姿态
    # θds = data["θds"]  # 期望的旋转角度
    num_configs = qs.shape[0]  # 测量配置数量。取qs这个元素个数，应该是一个列表，里面的测量数据的数量

    # 初始位姿的平移分量和旋转矩阵（平面部分）
    r_vw_w0 = q0[:2]  # 提取前两个元素：初始位置的 x 和 y 坐标
    C_wv0 = rotz(q0[2])[:2, :2]  # 初始旋转矩阵，取前两行两列，只考虑平面旋转

    # 构造最小二乘问题，用于找到不随旋转移动的基座位置
    A = np.zeros((2 * num_configs, 2))  # 系数矩阵
    b = np.zeros(2 * num_configs)  # 右侧常量向量

    # 遍历每个测量配置，构建线性系统
    for i in range(num_configs):
        r_vw_w = qs[i, :2]  # 当前测量姿态的位置部分
        C_wv = rotz(qs[i, 2])[:2, :2]  # 当前测量姿态的旋转矩阵

        # 填充最小二乘的系数矩阵和常量向量
        A[i * 2: i * 2 + 2, :] = C_wv - C_wv0  # 当前旋转矩阵的差
        b[i * 2: i * 2 + 2] = r_vw_w0 - r_vw_w  # 位置变化

    #! 具体推导见草稿纸，到时候应该push一个推导的PDF，把草稿纸的内容写入iPad上

    # 求解最小二乘问题，计算偏移量
    r_bv_v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # 使用零位姿的四元数，将偏移量旋转到基座坐标系
    #! 这个Q_bv是率先计算出来的(使用calibrate_base_position.py完成的)
    #! 上面的最小二乘结果是r_bv_v，这个r_bv_v是vicon坐标系下的偏移量
    #! 这里说明一下，此代码的符号表示：   world系，vicon系：marker贴出来的坐标系，base系：移动底盘的旋转中心
    #!          Q_bv: 是vicon坐标系相对于base坐标系的旋转；
    #!          r_bv_v: 是vicon坐标系相对于base坐标系的平移(错误)，应该是：base坐标系相对于vicon坐标系的平移，在vicon坐标系下的表示
    C_bv = q2r(Q_bv, order="xyzs")  # 将四元数转换为旋转矩阵
    #! 这里的C_bv是3*3的，所以r_bv_v要补充0，(-1*,是因为要求vicon相对于base的偏移，所以要指vico系)
    #! 故而：C_bv * np.append(r_bv_v, 0) ： 就是一个只有旋转的变换产生的齐次矩阵，求子坐标系下(旋转后的坐标系)的一个坐标在父坐标系下(对其进行了旋转)的表示
    r_vb_b = -C_bv @ np.append(r_bv_v, 0)  # 偏移量转换为基座坐标系

    # 输出结果，供后续校准使用
    print(f"r_vb_b = {r_vb_b} (使用此值作为 Vicon 零位姿的偏移量)") #! 表示vicon系相对于base系的平移


if __name__ == "__main__":
    main()
