import sys
from functools import partial
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt
from tongbot_hardware_central import BAG_DIR, ros_utils
import tongbot_hardware_central.system_identification as sysid


BAG_PATHS = [
    BAG_DIR + "/2022-07-25/velocity_test/" + path
    for path in [
        "velocity_test_joint0_v0.5_2022-07-25-15-32-43.bag",
        "velocity_test_joint1_v0.5_2022-07-25-15-39-15.bag",
        "velocity_test_joint2_v0.5_2022-07-25-15-46-14.bag",
        "velocity_test_joint3_v0.5_2022-07-25-15-55-46.bag",
        "velocity_test_joint4_v0.5_2022-07-25-16-01-38.bag",
        "velocity_test_joint5_v0.5_2022-07-25-16-04-08.bag",
    ]
] #? 使用时需要修改路径


class Step:
    def __init__(self, t0=0, magnitude=1):
        self.t0 = t0
        self.magnitude = magnitude

    def sample(self, ts, td=0):
        xs = np.zeros_like(ts)
        xs[ts >= self.t0 + td] = self.magnitude
        return xs


def parse_velocity_data(feedback_msgs, cmd_msgs, cmd_ts, idx):
    t0 = cmd_ts[0] - 0.1  # small extra buffer before commands start
    t1 = cmd_ts[-1] - 0.9
    feedback_msgs = ros_utils.trim_msgs(feedback_msgs, t0=t0, t1=t1)
    ts, _, vs = ros_utils.parse_ur10_joint_state_msgs(feedback_msgs)

    # TODO, ideally we'd make independent the range of messages and the ones
    # we'd like to use for fitting the curve

    # all of the commands should be the same
    cmd_value = cmd_msgs[0].data[idx]

    us = np.zeros_like(ts)
    for i in range(ts.shape[0]):
        us[i] = 0 if ts[i] < cmd_ts[0] else cmd_value

    step = Step(t0=cmd_ts[0] - ts[0], magnitude=cmd_value)

    # normalize time so that ts[0] = 0
    ts -= ts[0]

    # take only the velocity from joint idx
    vs = vs[:, idx]

    return ts, us, vs, step


def process_one_bag(path, joint_idx, save=False):
    bag = rosbag.Bag(path)

    # TODO these topic names are deprecated
    feedback_msgs = [
        msg for _, msg, _ in bag.read_messages("/my_gen3/joint_states")
    ]

    cmd_msgs = [msg for _, msg, _ in bag.read_messages("/my_gen3/cmd_vel")]

    # get times of the command messages, since these do not contain a timestamp
    cmd_ts = np.array(
        [t.to_sec() for _, _, t in bag.read_messages("/my_gen3/cmd_vel")]
    )

    ts, us, ys_actual, step = parse_velocity_data(
        feedback_msgs, cmd_msgs, cmd_ts, idx=joint_idx
    )

    # fit first- and second-order models to the data
    ωn, ζ, td = sysid.identify_second_order_system(ts, us, ys_actual, step)
    τ = sysid.identify_first_order_system(ts, us, ys_actual)
    print(ωn)
    print(ζ)
    print(td)

    # forward simulate the models
    ys_fit1 = sysid.simulate_first_order_system(ts, τ, us)

    ys_fit2 = sysid.simulate_second_order_system(ts, ωn, ζ, step.sample(ts, td=td))

    # make sure all steps are in positive direction for consistency
    if us[-1] < 0:
        us = -us
        ys_actual = -ys_actual
        ys_fit1 = -ys_fit1
        ys_fit2 = -ys_fit2

    fig = plt.figure()

    # actual output
    plt.plot(ts, ys_actual, label="Actual")

    # commands
    plt.plot(ts, us, linestyle="--", label="Commanded")

    # fitted models
    plt.plot(ts, ys_fit1, label="First-order model")
    plt.plot(ts, ys_fit2, label="Second-order model")

    plt.text(0.6, 0.4, f"First-order\nτ = {τ:.2f}")
    plt.text(0.6, 0.2, f"Second-order\nω = {ωn:.2f}\nζ = {ζ:.2f}")

    plt.xlabel("Time (s)")
    plt.ylabel("Joint velocity (rad/s)")
    plt.legend()
    plt.grid()
    plt.title(f"Joint {joint_idx} Velocity")

    if save:
        name = f"joint_{joint_idx}_velocity.png"
        fig.savefig(name)
        print(f"Saved plot to {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save the plots to files.")
    args = parser.parse_args()

    for i, path in enumerate(BAG_PATHS):
        process_one_bag(path, i, save=args.save)
        # break
    plt.show()


if __name__ == "__main__":
    main()
