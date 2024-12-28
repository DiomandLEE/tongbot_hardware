#!/usr/bin/env python3
"""Tune Kalman filter for estimation of robot state."""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from tongbot_hardware_central import ros_utils
from tongbot_hardware_analysis.parsing import parse_mpc_observation_msgs

import IPython

'''
在测试移动机械臂的卡尔曼滤波状态估计，也是使用了central中的ros_utils的插值完成时间对齐
measured是robot发布的joint_state
integrated是MPC观测的速度积分，就是mpc的observation，但是感觉就是和measured差不多
estimated是卡尔曼滤波的结果
'''


ROBOT_PROC_VAR = 1000
ROBOT_MEAS_VAR = 0.001


class GaussianEstimate:
    def __init__(self, x, P):
        self.x = x
        self.P = P


def kf_predict(estimate, A, Q, v):
    x = A @ estimate.x + v
    P = A @ estimate.P @ A.T + Q
    return GaussianEstimate(x, P)


def kf_correct(estimate, C, R, y):
    # Innovation covariance
    CP = C @ estimate.P
    S = CP @ C.T + R

    # Correct using measurement model
    P = estimate.P - CP.T @ np.linalg.solve(S, CP)
    x = estimate.x + CP.T @ np.linalg.solve(S, y - C @ estimate.x)
    return GaussianEstimate(x, P)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)

    kinova_msgs = [msg for _, msg, _ in bag.read_messages("/my_gen3/joint_states")]
    dingo_msgs = [msg for _, msg, _ in bag.read_messages("/dingo/joint_states")]
    mpc_obs_msgs = [
        msg for _, msg, _ in bag.read_messages("/mobile_manipulator_mpc_observation")
    ] #todo 需要找一下在tongbot_mpc中这个的topic名称是什么

    tas, qas, vas = ros_utils.parse_kinova_joint_state_msgs(
        kinova_msgs, normalize_time=False
    )
    tbs, qbs, vbs = ros_utils.parse_dingo_joint_state_msgs(
        dingo_msgs, normalize_time=False
    )
    tms, xms, ums = parse_mpc_observation_msgs(mpc_obs_msgs, normalize_time=False)

    n = 10

    ts = tas
    qbs_aligned = ros_utils.interpolate_list(ts, tbs, qbs)
    vbs_aligned = ros_utils.interpolate_list(ts, tbs, vbs)

    # observations
    qs = np.hstack((qbs_aligned, qas))
    vs = np.hstack((vbs_aligned, vas))

    # inputs
    us = np.array(ros_utils.interpolate_list(ts, tms, ums))

    # integrated state (assumes perfect model)
    xms_aligned = np.array(ros_utils.interpolate_list(ts, tms, xms))
    vs_int = xms_aligned[:, n : 2 * n]

    ts -= ts[0]

    # initial state
    x0 = np.concatenate((qs[0, :], np.zeros(2 * n)))
    estimate = GaussianEstimate(x0, np.eye(x0.shape[0]))

    I = np.eye(n)
    Z = np.zeros((n, n))
    C = np.hstack((I, Z, Z))

    # noise covariance
    Q0 = ROBOT_PROC_VAR * I
    R = ROBOT_MEAS_VAR * I

    # do estimation using the Kalman filter
    xs_est = [x0]
    for i in range(1, ts.shape[0]):
        dt = ts[i] - ts[i - 1]

        A = np.block([[I, dt * I, 0.5 * dt * dt * I], [Z, I, dt * I], [Z, Z, I]])
        B = np.vstack((dt * dt * dt * I / 6, 0.5 * dt * dt * I, dt * I))
        Q = B @ Q0 @ B.T
        u = us[i, :n]

        estimate = kf_predict(estimate, A, Q, B @ u)
        estimate = kf_correct(estimate, C, R, qs[i, :])
        xs_est.append(estimate.x.copy())

    xs_est = np.array(xs_est)
    qs_est = xs_est[:, :n]
    vs_est = xs_est[:, n : 2 * n]

    plt.figure()
    for i in range(n):
        plt.plot(ts, qs[:, i], label=f"q_{i+1}")
    plt.title("Measured Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, qs_est[:, i], label=f"q_{i+1}")
    plt.title("Estimated Joint Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs[:, i], label=f"v_{i+1}")
    plt.title("Measured Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs_est[:, i], label=f"v_{i+1}")
    plt.title("Estimated Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs_int[:, i], label=f"v_{i+1}")
    plt.title("Integrated Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs[:, i] - vs_est[:, i], label=f"v_{i+1}")
    plt.title("Measured - Estimated Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(n):
        plt.plot(ts, vs[:, i] - vs_int[:, i], label=f"v_{i+1}")
    plt.title("Measured - Integrated Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
