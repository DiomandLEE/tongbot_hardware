#!/usr/bin/env python3
"""使用 `collect_arm_calibration_data.py` 收集的数据对手臂进行标定。

我们将问题形式化为在两个 SE(3) 流形的乘积上进行非线性优化。
"""
import argparse  # 导入用于命令行参数解析
import datetime  # 导入日期和时间模块，用于生成时间戳

import numpy as np  # 导入NumPy库用于数值计算
import pymanopt  # 导入pymanopt库用于流形优化
import yaml  # 导入yaml库用于读取和写入YAML文件

import jax  # 导入jax库用于自动微分
import jaxlie  # 导入jaxlie库用于处理SE(3)和SO(3)流形

# 导入自定义的运动学库，用于计算机器人的正向运动学
from tongbot_hardware_central import MobileManipulatorKinematics


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "data_file_name", help="NPZ 文件，包含标定数据。"  # 第一个参数：标定数据文件
    # )
    # parser.add_argument(
    #     "-o",
    #     "--output_file_name",
    #     help="输出的 YAML 文件，用于保存优化后的变换矩阵。"  # 第二个参数：输出文件名
    # )
    parser.add_argument(
        "data_file_name", help="NPZ file containing the calibration data."
    )
    parser.add_argument(
        "-o",
        "--output_file_name",
        help="YAML file to output with optimized transforms.",
    )
    args = parser.parse_args()  # 解析命令行参数

    # qs = []  # joint configurations
    # Qs = []  # object orientation quaternions
    # rs = []  # object translations

    # 加载标定数据
    data = np.load(args.data_file_name)  # 从NPZ文件加载数据
    qs = data["qs"]  # 获取关节配置q-joint states（`qs`）
    # qds = data["qds"]  # 注释掉的行，可能用于关节速度数据
    r_tw_ws_meas = data["rs"]  # 获取测量的gripper抓取的object位置（transation）
    Q_wts_meas = data["Qs"]  # 获取测量的gripper抓取的object姿态（四元数）

    num_configs = qs.shape[0]  # 获取标定数据中关节配置的数量

    # 计算每个q对应的base位姿
    T_wbs = []  # wTb，frame base到world的变换s
    for i in range(num_configs):
        # qs就相当于是所有自由度的q，这下q0和q1就是base的x和y坐标
        r_bw_w = np.array([qs[i, 0], qs[i, 1], 0])
        C_wb = jaxlie.SO3.from_z_radians(qs[i, 2])  # 计算base的旋转矩阵（以 z 轴为旋转轴）
        T_wb = jaxlie.SE3.from_rotation_and_translation(
            C_wb, r_bw_w)  # 组合旋转和平移，得到SE(3)变换，齐次变换矩阵4*4
        T_wbs.append(T_wb)  # 将计算的变换添加到列表中

    # 计算tool到的测量变换（反向变换）
    T_tws_meas = []  # tTw，frame world到tool的变换s，s表示多个
    for i in range(num_configs):
        C_wt_meas = jaxlie.SO3.from_quaternion_xyzw(
            Q_wts_meas[i, :])  # 计算工具的旋转矩阵（四元数转换）
        T_wt_meas = jaxlie.SE3.from_rotation_and_translation(
            C_wt_meas, r_tw_ws_meas[i, :]  # 计算wTt， frame tool到world的变换
        )
        T_tws_meas.append(T_wt_meas.inverse())  # 求逆变换，得到从工具到基座的变换

    # 使用运动学模型计算基座到工具的模型化变换
    # 创建机器人运动学模型实例，加载urdf，并添加三个VirtualJoint
    kinematics = MobileManipulatorKinematics()
    T_bts_model = []  # bTt，frame tool到base的变换s
    for i in range(num_configs):
        kinematics.forward(qs[i, :])  # pinocchio中的FK and updatePlace
        # 返回gripper(构造函数中默认的)frame在world中的位置和姿态
        r_tw_w_model, Q_wt_model = kinematics.link_pose()
        C_wt_model = jaxlie.SO3.from_quaternion_xyzw(Q_wt_model)  # 将四元数转化为旋转矩阵
        T_wt_model = jaxlie.SE3.from_rotation_and_translation(
            C_wt_model, r_tw_w_model)  # 将旋转矩阵和平移向量组合，得到SE(3)变换
        # 计算从基座到工具的模型化变换，并与测量值进行比较
        # T_wb的逆矩阵乘以T_wt，得到T_bt，frame tool到base的变换
        T_bts_model.append(T_wbs[i].inverse() @ T_wt_model)

    # 定义优化流形：优化两个SE(3)变换
    manifold = pymanopt.manifolds.Product(
        (
            pymanopt.manifolds.SpecialOrthogonalGroup(3),  # SO(3)流形，旋转部分
            pymanopt.manifolds.Euclidean(3),  # 3D平移流形，Euclidean(3)，平移部分
            pymanopt.manifolds.SpecialOrthogonalGroup(
                3),  # 第二个SO(3)，
            pymanopt.manifolds.Euclidean(3),  # 第二个Euclidean(3)
        )
    )

    # 定义从旋转矩阵和平移向量构造SE(3)变换的函数
    def transform(C, r):
        return jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3.from_matrix(C), r)

    # 定义将SE(3)变换转化为字典的函数，方便保存到YAML
    def transform_dict(T):
        return {
            "xyz": list([float(x) for x in T.translation()]),  # 提取平移部分并转化为列表
            # 提取旋转部分，并转换为RPY（欧拉角）表示
            "rpy": list([float(x) for x in T.rotation().as_rpy_radians()]),
        }

    # jax版本的代价函数，用于自动微分
    def jcost(C1, r1, C2, r2):
        T1 = transform(C1, r1)  # base到arm_base的变换
        T2 = transform(C2, r2)  # frame_tool到手grippr的变换

        cost = 0  # 初始化代价
        for i in range(num_configs):
            # 计算当前配置下的变换误差
            ΔT = T_wbs[i] @ T1 @ T_bts_model[i] @ T2 @ T_tws_meas[i]
            # world T base * T1 *  base_arm T grippr * T2 @ tTw @ wTw
            e = ΔT.log()  # ! 矩阵取对数操作 (log) 通常适用于 特殊正交群 (SO(3)) 或 特殊欧氏群SE(3)) 中的变换矩阵。
            #! 这种操作被称为Lie对数映射 (Logarithmic Map)，它将一个矩阵形式的变换映射到其对应的李代数表示（一个向量或矩阵）
            #! 矩阵的误差越小，越趋近于单位阵， 取对数后，误差向量e的模长越小，越趋近于0
            cost = cost + 0.5 * e @ e  # 累加误差的平方
        return cost

    # 计算代价函数的梯度
    # ! argnums指定要计算梯度的参数，0，1，2，3分别对应C1, r1, C2, r2
    jgrad = jax.grad(jcost, argnums=(0, 1, 2, 3))

    #! : 将一个普通的 Python 函数转换成一个可以在 pymanopt 中使用的函数。这个装饰器的主要目的是使函数适应流形优化方法。
    # pymanopt中的代价函数，包装了jax计算的代价函数
    @pymanopt.function.numpy(manifold)
    def cost(C1, r1, C2, r2):
        return jcost(C1, r1, C2, r2)

    # pymanopt中的梯度函数，包装了jax计算的梯度
    @pymanopt.function.numpy(manifold)
    def gradient(C1, r1, C2, r2):
        return jgrad(C1, r1, C2, r2)

    # 设置初始猜测
    C20 = np.array([[0.0, 0, 1], [0, -1, 0], [1, 0, 0]])  # 初始猜测的旋转矩阵
    r20 = np.array([0, 0, 0.3])  # 初始猜测的平移向量
    x0 = (np.eye(3), np.zeros(3), C20, r20)  # 初始猜测的参数

    # 设置并求解优化问题
    problem = pymanopt.Problem(
        manifold, cost, euclidean_gradient=gradient)  # 创建优化问题
    line_searcher = pymanopt.optimizers.line_search.BackTrackingLineSearcher()  # 线性搜索方法
    optimizer = pymanopt.optimizers.SteepestDescent(
        line_searcher=line_searcher)  # 使用梯度下降法进行优化
    result = optimizer.run(problem, initial_point=x0)  # 执行优化
    #! 以上是一个固定的套路，记住就好

    # 获取优化后的变换
    T1_opt = transform(result.point[0], result.point[1])  # 基座到手臂的优化变换
    T2_opt = transform(result.point[2], result.point[3])  # 工具到手臂的优化变换

    # 将优化结果保存为字典
    yaml_dict = {
        "base_to_arm_transform": transform_dict(T1_opt),
        "gripped_object_transform": transform_dict(T2_opt),
    }

    print(yaml.dump(yaml_dict))  # 打印优化后的变换结果

    # 如果指定了输出文件名，则将
    #! 实际执行的时候，把tool换成，二指夹爪的一个爪子。
