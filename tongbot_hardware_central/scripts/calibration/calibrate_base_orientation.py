#!/usr/bin/env python3
"""
使用 `collect_base_orientation_calibration_data.py` 收集的数据校准底座。
方法：
让底盘沿者x轴方向移动，由于要是vicon上base的x轴与world系上的x轴是重合的话，是没有偏移的
若是没有偏移的话，那么在是不会在y方向上有变化的
"""
# 导入必要的库
import argparse  # 用于解析命令行参数
import numpy as np  # 用于数值计算
from spatialmath.base import rotz, r2q  # 空间数学库，用于计算旋转矩阵和四元数

# 主函数
def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="包含校准数据的 npz 文件。")
    args = parser.parse_args()

    # 加载校准数据
    # 跳过第一个位置（起点）
    data = np.load(args.filename)  # 加载保存的 npz 数据文件
    q0 = data["q0"]  # 初始配置
    qs = data["qs"][1:, :]  # 跳过第一个配置
    xds = data["xds"]  # 目标位置序列
    num_configs = qs.shape[0]  # 测试的配置数量

    # 构造最小二乘问题：寻找最佳的角度偏移量，用于修正底座框架与 Vicon 模型框架之间的方向误差
    A = np.ones((num_configs, 1))  # 设计矩阵，每行都是 1
    b = np.zeros(num_configs)  # 观测值向量
    for i in range(num_configs):
        yaw = qs[i, 2] - q0[2]  # 当前vicon配置相对于初始vicon配置的偏航角
        C_bw = rotz(yaw)[:2, :2].T  # 旋转矩阵，表示从vicon_i到vicon_0的变换
        #! .T是转置，但是对于旋转矩阵(正交的)，转置和逆矩阵是一样的，故而表示从vicon_0到vicon_i的变换
        r_w = qs[i, :2] - q0[:2]  # vicon_0坐标系中的平移向量
        r_b = C_bw @ r_w  # 转换到vicon_i系的平移向量，没有* 负号，说明就是计算vicon_0中的向量，在vicon_i中的表示
        angle = np.arctan2(r_b[1], r_b[0])  # 计算角度偏移
        b[i] = angle  # 存储角度偏移
        #! A矩阵就是一个全1的矩阵，b矩阵存储的是角度偏移
        print(angle)  # 打印每个配置对应的角度偏移

    # 求解最佳偏移量
    Δθ, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # 最小二乘求解
    assert Δθ.size == 1  # 确保结果是一个标量

    # 得到的Δθ是C_vb的偏移量，即real_base_frame和 vicon_base_frame的角度偏移量，C_bv
    # 从real_base_frame到vicon_base_frame的变换是C_vb

    # 负号用于匹配 Vicon 系统的零位姿约定。#! 错误！！！，我认为是错的，需要真机试一下
    # 原始值对应于 C_bv。为了达到 Vicon 系统的 *零位姿*，需要旋转 C_bv 来移除方向偏移。
    C_vb = rotz(-Δθ[0])  # 计算修正旋转矩阵 #! 得到的是从base系到vicon系的旋转矩阵(坐标转换矩阵中的R)，base系相对于vicon旋转的R,-delta
    Q_vb = r2q(C_vb, order="xyzs")  # 将旋转矩阵转换为四元数

    C_bv = rotz(Δθ[0])  #! 得到的是从vicon系到base系的旋转矩阵(坐标转换矩阵中的R)，vicon系相对于base旋转的R
    Q_bv = r2q(C_bv, order="xyzs")  # 将旋转矩阵转换为四元数

    #todo 我认为是这样的，原始代码是有错的，delta是表示vicon系相对于base系的旋转，到时候测试一下，故意把dingo的marker逆时针贴歪
    # 打印结果
    print(f"最佳底座方向偏移量 = {Δθ}")
    print(f"Q_bv = {Q_bv}")
    print("这对应于 Vicon 的零位姿约定。")

# 程序入口
if __name__ == "__main__":
    main()
