#!/usr/bin/env python3
"""Plot all figures from data, the same as after a simulation run."""
import sys
import numpy as np

from tongbot_hardware_analysis.logging import DataPlotter


def main():
    plotter = DataPlotter.from_npz(sys.argv[1])
    plotter.plot_all(show=True)


if __name__ == "__main__":
    main()
