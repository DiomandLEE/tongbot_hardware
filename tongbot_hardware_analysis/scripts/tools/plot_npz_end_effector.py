#!/usr/bin/env python3
"""Plot the state x over time."""
import sys
import numpy as np

import tongbot_hardware_analysis.logging as logging
import tongbot_hardware_analysis.math as tmath



def main():
    #： python plot.py data.npz
    plotter = logging.DataPlotter.from_npz(sys.argv[1])
    plotter.plot_ee_position()
    plotter.plot_ee_orientation()
    plotter.plot_ee_velocity()

    # compute final tilt angle (we want this for the wedge simulations)
    Qf = plotter.data["Q_wes"][-1, :]
    z0 = np.array([0, 0, 1])
    zf = tmath.quat_rotate(Qf, z0)
    angle = np.arccos(z0 @ zf)
    print(f"Final tilt angle = {np.rad2deg(angle)} degrees")

    plotter.show()


if __name__ == "__main__":
    main()
