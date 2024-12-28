#!/usr/bin/env python3
"""Plot the state x over time."""
import sys
import numpy as np

from tongbot_hardware_analysis.logging import DataPlotter

#! 这个npz的脚本，需要对应的npz文件，而对应的文件，在upright中是通过mpc_sim.py得到的，我们没有自己写，所以这些脚本暂时用不到

def main():
    plotter = DataPlotter.from_npz(sys.argv[1])
    plotter.plot_state()
    plotter.show()


if __name__ == "__main__":
    main()
