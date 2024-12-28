# #!/usr/bin/env python3
# """Generate and save a trajectory by rolling out the MPC."""
# import numpy as np

# import upright_core as core
# import upright_control as ctrl
# import upright_cmd as cmd

# import IPython


# def main():
#     np.set_printoptions(precision=3, suppress=True)

#     argparser = cmd.cli.basic_arg_parser()
#     argparser.add_argument("trajectory_file", help="NPZ file to save the trajectory to.")
#     cli_args = argparser.parse_args()

#     # load configuration
#     config = core.parsing.load_config(cli_args.config)
#     sim_config = config["simulation"]
#     ctrl_config = config["controller"]

#     timestep = sim_config["timestep"]
#     duration = sim_config["duration"]

#     # rollout the controller to generate a trajectory
#     ctrl_manager = ctrl.manager.ControllerManager.from_config(ctrl_config)
#     #! 就是求解了一次得到了一个轨迹，step函数是求解一次，plan函数是求解多次，得到一个轨迹
#     # ../upright/upright_control/src/upright_control/manager.py
#     trajectory = ctrl_manager.plan(timestep, duration)
#     trajectory.save(cli_args.trajectory_file)

#     print(f"Saved trajectory to {cli_args.trajectory_file}.")


# if __name__ == "__main__":
#     main()
