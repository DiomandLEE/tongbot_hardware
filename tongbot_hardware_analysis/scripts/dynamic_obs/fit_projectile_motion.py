#!/usr/bin/env python3
"""Plot position of Vicon objects from a ROS bag.

Also fit the data with projectile motion models and filter with a Kalman filter.
"""
'''
这段代码以实验数据（真实的vicon数据）为基础，通过空气阻力运动模型的前向仿真和参数识别，构建一个完整的运动分析工具链。

核心目标：
通过运动模型重现物体的真实运动。
自动化识别空气阻力系数。
提供平滑的速度估计以支持进一步分析。
'''
#! 本实验用不到
import argparse
from scipy import optimize

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from tongbot_hardware_central import ros_utils

import IPython


# we only use data above this height
H = 0.1


def rollout_drag(ts, b, r0, v0, g):
    """Roll out model including a nonlinear drag term with coefficient b."""
    n = ts.shape[0]
    rs = np.zeros((n, 3))
    vs = np.zeros((n, 3))

    rs[0, :] = r0
    vs[0, :] = v0

    for i in range(1, n):
        dt = ts[i] - ts[i - 1]
        a = g - b * np.linalg.norm(vs[i - 1, :]) * vs[i - 1, :]
        vs[i, :] = vs[i - 1, :] + dt * a
        rs[i, :] = rs[i - 1, :] + dt * vs[i, :]
    return rs, vs
'''
\text{Given: } \mathbf{r}_0, \mathbf{v}_0, \mathbf{g}, \mathbf{b}, \text{ and } \mathbf{t} \text{ (time vector)}.

\text{For each time step } t_i:
\begin{align*}
\Delta t &= t_i - t_{i-1}, \\
\mathbf{a}(t_i) &= \mathbf{g} - b \|\mathbf{v}(t_{i-1})\| \mathbf{v}(t_{i-1}), \\
\mathbf{v}(t_i) &= \mathbf{v}(t_{i-1}) + \Delta t \cdot \mathbf{a}(t_i), \\
\mathbf{r}(t_i) &= \mathbf{r}(t_{i-1}) + \Delta t \cdot \mathbf{v}(t_i).
\end{align*}
模拟一个带有阻力的抛物运动模型。
b是阻尼系数，阻力是非线性的
'''


def identify_drag(ts, rs, r0, v0, g, method="trf", p0=[0.01]):
    """Fit a second-order model to the inputs us and outputs ys at times ts."""

    def residuals(b):
        rm, _ = rollout_drag(ts, b, r0, v0, g)
        Δs = np.linalg.norm(rm - rs, axis=1)
        return Δs

    bounds = ([0], [np.inf])
    res = optimize.least_squares(residuals, x0=p0, bounds=bounds)
    return res.x[0]
'''
\text{Given: } \mathbf{t}, \mathbf{r}(t), \mathbf{r}_0, \mathbf{v}_0, \mathbf{g}.

\text{Objective: Minimize the residual error: }
\begin{align*}
\mathbf{\Delta}_s(b) &= \|\mathbf{r}_{\text{measured}}(t) - \mathbf{r}_{\text{model}}(t, b)\|.
\end{align*}

\text{Residual error is minimized using: }
\begin{align*}
\hat{b} = \underset{b}{\arg\min} \sum_{i=1}^N \|\mathbf{r}_{\text{measured}}(t_i) - \mathbf{r}_{\text{model}}(t_i, b)\|^2,
\end{align*}
\text{where } \mathbf{r}_{\text{model}}(t, b) \text{ is computed using } \texttt{rollout\_drag}.
通过最小化残差来识别阻力系数
逻辑：
定义残差函数：实际位置与带阻力模型预测位置之间的差距：
'''

def rollout_numerical_diff(ts, rs, r0, v0, τ):
    """Rollout velocity using exponential smoothing."""
    n = ts.shape[0]
    vs = np.zeros((n, 3))
    vs[0, :] = v0

    for i in range(1, n):
        dt = ts[i] - ts[i - 1]
        α = 1 - np.exp(-dt / τ)
        v_meas = (rs[i, :] - rs[i - 1, :]) / dt
        vs[i, :] = α * v_meas + (1 - α) * vs[i - 1, :]
    return vs
'''
\text{Given: } \mathbf{t}, \mathbf{r}(t), \mathbf{v}_0, \tau.

\text{For each time step } t_i:
\begin{align*}
\Delta t &= t_i - t_{i-1}, \\
\alpha &= 1 - \exp\left(-\frac{\Delta t}{\tau}\right), \\
\mathbf{v}_{\text{meas}}(t_i) &= \frac{\mathbf{r}(t_i) - \mathbf{r}(t_{i-1})}{\Delta t}, \\
\mathbf{v}(t_i) &= \alpha \cdot \mathbf{v}_{\text{meas}}(t_i) + (1 - \alpha) \cdot \mathbf{v}(t_{i-1}).
\end{align*}
利用指数平滑对速度进行数值微分。
此时刻的速度 = 此时刻的diff速度 与 上一时刻的速度的加权平均
'''


def rollout_kalman(ts, rs, r0, v0, g):
    """Estimate ball trajectory using a Kalman filter."""
    n = ts.shape[0]

    # noise covariance
    dt_nom = 0.01
    R = dt_nom**2 * np.eye(3)

    # acceleration variance
    var_a = 1000

    # (linear) motion model
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    C = np.zeros((3, 6))
    C[:, :3] = np.eye(3)

    # initial state
    xs = np.zeros((n, 6))
    xc = np.concatenate((r0, v0))
    Pc = np.eye(6)  # P0

    xs[0, :] = xc
    Ps = [Pc]

    active = False

    for i in range(1, n):
        dt = ts[i] - ts[i - 1]

        Ai = np.eye(6) + dt * A
        Bi = np.vstack((0.5 * dt**2 * np.eye(3), dt * np.eye(3)))

        # NOTE: doing this is key! (rather than having no off-diagonal elements)
        Qi = var_a * Bi @ Bi.T
        Ri = R

        if rs[i, 2] >= 0.8:
            active = True
        elif rs[i, 2] <= 0.2:
            active = False

        if active:
            u = g
        else:
            u = np.zeros(3)

        # predictor
        xc = Ai @ xs[i - 1, :] + Bi @ u
        Pc = Ai @ Ps[i - 1] @ Ai.T + Qi

        # measurement
        y = rs[i, :]

        # corrector
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

    msgs = [msg for _, msg, _ in bag.read_messages("/vicon/ThingVolleyBall/ThingVolleyBall")]
    positions = []
    for msg in msgs:
        p = msg.transform.translation
        positions.append([p.x, p.y, p.z])
    positions = np.array(positions)
    times = ros_utils.parse_time(msgs)

    # find portion of the ball undergoing projectile motion
    idx = np.flatnonzero(positions[:, 2] >= H)
    rp = positions[idx, :]
    tp = times[idx]

    # use the first two timesteps to estimate initial state
    r0 = rp[1, :]
    v0 = (rp[1, :] - rp[0, :]) / (tp[1] - tp[0])
    g = np.array([0, 0, -9.81])

    # discard first timestep now that we've "used it up"
    rp = rp[1:, :]
    tp = tp[1:]

    # nominal model (perfect projectile motion)
    tm = (tp - tp[0])[:, None]
    rm = r0 + v0 * tm + 0.5 * tm ** 2 * g
    vm = v0 + tm * g

    # drag model
    b = identify_drag(tp, rp, r0, v0, g)
    print(f"b = {b}")
    rd, vd = rollout_drag(tp, b, r0, v0, g)

    # numerical diff to get velocity
    vn = rollout_numerical_diff(tp, rp, r0, v0, τ=0.05)

    # rollout with Kalman filter
    xk, Pks = rollout_kalman(tp, rp, r0, v0, g)
    rk, vk = xk[:, :3], xk[:, 3:]

    # Position models

    plt.figure()

    # ground truth
    plt.plot(tp, rp[:, 0], label="x (vicon)", color="r")
    plt.plot(tp, rp[:, 1], label="y (vicon)", color="g")
    plt.plot(tp, rp[:, 2], label="z (vicon)", color="b")

    # assume perfect projectile motion
    plt.plot(tp, rm[:, 0], label="x (nominal)", color="r", linestyle=":")
    plt.plot(tp, rm[:, 1], label="y (nominal)", color="g", linestyle=":")
    plt.plot(tp, rm[:, 2], label="z (nominal)", color="b", linestyle=":")

    # Kalman filter (no drag)
    plt.plot(tp, rk[:, 0], label="x (kalman)", color="r", linestyle="--")
    plt.plot(tp, rk[:, 1], label="y (kalman)", color="g", linestyle="--")
    plt.plot(tp, rk[:, 2], label="z (kalman)", color="b", linestyle="--")

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Projectile position vs. time")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(tp, Pks[:, 0, 0], label="var(x)", color="r")
    plt.plot(tp, Pks[:, 3, 3], label="var(vx)", color="r", linestyle="--")
    plt.legend()
    plt.title("KF variance")
    plt.grid()

    # Position model with drag

    plt.figure()

    # ground truth
    plt.plot(tp, rp[:, 0], label="x (vicon)", color="r")
    plt.plot(tp, rp[:, 1], label="y (vicon)", color="g")
    plt.plot(tp, rp[:, 2], label="z (vicon)", color="b")

    # assume perfect projectile motion
    plt.plot(tp, rm[:, 0], label="x (nominal)", color="r", linestyle=":")
    plt.plot(tp, rm[:, 1], label="y (nominal)", color="g", linestyle=":")
    plt.plot(tp, rm[:, 2], label="z (nominal)", color="b", linestyle=":")

    # projectile motion with drag
    plt.plot(tp, rd[:, 0], label="x (drag)", color="r", linestyle="-.")
    plt.plot(tp, rd[:, 1], label="y (drag)", color="g", linestyle="-.")
    plt.plot(tp, rd[:, 2], label="z (drag)", color="b", linestyle="-.")

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Projectile position vs. time: drag")
    plt.legend()
    plt.grid()

    # Velocity models

    plt.figure()

    # numerically diffed model
    plt.plot(tp, vn[:, 0], label="x (num diff)", color="r")
    plt.plot(tp, vn[:, 1], label="y (num diff)", color="g")
    plt.plot(tp, vn[:, 2], label="z (num diff)", color="b")

    # perfect projectile motion
    plt.plot(tp, vm[:, 0], label="x (nominal)", color="r", linestyle=":")
    plt.plot(tp, vm[:, 1], label="y (nominal)", color="g", linestyle=":")
    plt.plot(tp, vm[:, 2], label="z (nominal)", color="b", linestyle=":")

    # kalman filter
    plt.plot(tp, vk[:, 0], label="x (kalman)", color="r", linestyle="--")
    plt.plot(tp, vk[:, 1], label="y (kalman)", color="g", linestyle="--")
    plt.plot(tp, vk[:, 2], label="z (kalman)", color="b", linestyle="--")

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Projectile velocity vs. time")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
