#!/usr/bin/env python3
"""Play with KF for projectile motion."""
'''
还是后处理，处理真实vicon数据
'''
import argparse
from scipy import optimize

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from tongbot_hardware_central import ros_utils

import IPython


def rollout_numerical_diff(ts, rs, r0, v0, τ):
    """Rollout velocity using exponential smoothing."""
    n = ts.shape[0]
    vs = np.zeros((n, 2))
    vs[0, :] = v0

    for i in range(1, n):
        dt = ts[i] - ts[i - 1]
        α = 1 - np.exp(-dt / τ)
        v_meas = (rs[i, :] - rs[i - 1, :]) / dt
        vs[i, :] = α * v_meas + (1 - α) * vs[i - 1, :]
    return vs


def rollout_kalman(ts, rs, r0, v0, g):
    """Estimate ball trajectory using a Kalman filter."""
    n = ts.shape[0] #取出ts的行数，其实就是1

    # noise covariance
    dt_nom = 0.01
    R = dt_nom**2 * np.eye(2)  #1e-4

    # acceleration variance
    var_a = 1000

    # (linear) motion model
    A = np.zeros((6, 6))

    C = np.array([[1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0]])

    # initial state
    xs = np.zeros((n, 6))
    xc = np.concatenate((r0, v0, [0, 0])) #用于连接多个数组的函数，axis=0表示按行连接(行数不变)，axis=1表示按列连接(列数不变)
    Pc = np.diag([1, 1, 1, 1, 10, 10])  # P0

    xs[0, :] = xc
    Ps = [Pc]

    active = False

    for i in range(1, n):
        dt = ts[i] - ts[i - 1]

        Ai = np.array([[1, 0, dt, 0, 0.5 * dt**2, 0],
                           [0, 1, 0, dt, 0, 0.5 * dt**2],
                            [0, 0, 1, 0, dt, 0],
                            [0, 0, 0, 1, 0, dt],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])
        # Bi = np.vstack((0.5 * dt**2 * np.eye(3), dt * np.eye(3))) #垂直堆叠，列数不变，行数增加

        # NOTE: doing this is key! (rather than having no off-diagonal elements)
        Qi = 1e-3 * np.array([[dt**4 / 4, 0, dt**3 / 2, 0, dt**2 / 2, 0],
                        [0, dt**4 / 4, 0, dt**3 / 2, 0, dt**2 / 2],
                        [dt**3 / 2, 0, dt**2, 0, dt, 0],
                        [0, dt**3 / 2, 0, dt**2, 0, dt],
                        [dt**2 / 2, 0, dt, 0, 1, 0],
                        [0, dt**2 / 2, 0, dt, 0, 1]])

        Ri = R

        # predictor
        xc = Ai @ xs[i - 1, :]
        Pc = Ai @ Ps[i - 1] @ Ai.T + Qi

        # measurement
        y = rs[i, :]

        # corrector
        # np.linalg.inv(C @ Pc @ C.T + Ri) 是 NumPy 中的一种矩阵操作，表示计算矩阵
        # C⋅Pc⋅CT+Ri 的逆矩阵。其中，C 是测量矩阵，Pc 是预测误差协方差矩阵，Ri 是测量噪声协方差矩阵。
        K = Pc @ C.T @ np.linalg.inv(C @ Pc @ C.T + Ri)
        P = (np.eye(6) - K @ C) @ Pc
        xs[i, :] = xc + K @ (y - C @ xc)
        Ps.append(P)

    return xs, np.array(Ps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)

    msgs = [
        msg for _, msg, _ in bag.read_messages("/vicon/Dynamic_Obs/Dynamic_Obs")
    ]
    positions = []
    for msg in msgs:
        p = msg.transform.translation
        positions.append([p.x, p.y])
    positions = np.array(positions)
    times = ros_utils.parse_time(msgs)

    # find portion of the ball undergoing projectile motion
    # 这些索引是以一维数组形式返回的，即使原始数组是多维的。将原始数组展开为1维数组，然后返回非零元素的索引。
    idx = np.flatnonzero(positions[:, 1] >= -1000)
    rp = positions[idx, :]
    tp = times[idx]

    # use the first two timesteps to estimate initial state
    r0 = rp[1, :]
    v0 = (rp[1, :] - rp[0, :]) / (tp[1] - tp[0]) #使用开始的两帧数据的差代表初始
    g = np.array([0, 0, -9.81])

    # discard first timestep now that we've "used it up"
    rp = rp[1:, :] #measured
    tp = tp[1:]

    # numerical diff to get velocity
    vn = rollout_numerical_diff(tp, rp, r0, v0, τ=0.45)

    # rollout with Kalman filter
    xk, Pks = rollout_kalman(tp, rp, r0, v0, g)
    rk, vk = xk[:, :2], xk[:, 2:4]

    # Position models

    plt.figure()

    # ground truth
    plt.plot(tp, rp[:, 0], label="x (vicon)", color="r")
    plt.plot(tp, rp[:, 1], label="y (vicon)", color="g")

    # Kalman filter (no drag)
    plt.plot(tp, rk[:, 0], label="x (kalman)", color="r", linestyle="--")
    plt.plot(tp, rk[:, 1], label="y (kalman)", color="g", linestyle="--")

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Projectile position vs. time")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(tp, Pks[:, 0, 0], label="var(x)", color="r")
    plt.plot(tp, Pks[:, 2, 2], label="var(vx)", color="r", linestyle="--")
    plt.legend()
    plt.grid()

    # Velocity models
    plt.figure()

    # numerically diffed model
    plt.plot(tp, vn[:, 0], label="x (num diff)", color="r")
    plt.plot(tp, vn[:, 1], label="y (num diff)", color="g")

    # kalman filter
    plt.plot(tp, vk[:, 0], label="x (kalman)", color="r", linestyle="--")
    plt.plot(tp, vk[:, 1], label="y (kalman)", color="g", linestyle="--")

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Projectile velocity vs. time")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
