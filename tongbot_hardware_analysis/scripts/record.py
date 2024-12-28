#!/usr/bin/env python3
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import signal
import time
import yaml

import IPython


ROSBAG_CMD_ROOT = ["rosbag", "record"]
# fmt: off
ROSBAG_TOPICS = [
        "/clock",
        "--regex", "/ridgeback/(.*)",
        "--regex", "/ridgeback_velocity_controller/(.*)",
        "--regex", "/ur10/(.*)",
        "--regex", "/vicon/(.*)",
        "--regex", "/projectile/(.*)",
        "--regex", "/mobile_manipulator_(.*)"
]
# fmt: on

# ROSBAG_CMD_ROOT：定义了启动 rosbag 记录的基础命令。这里是 rosbag record，它会记录指定的 ROS 主题。
# ROSBAG_TOPICS：列出了要记录的主题。使用了 --regex 参数来指定一个正则表达式，
#   匹配以某些特定前缀开头的主题（例如 /ridgeback/、/ur10/ 等）。这意味着记录的主题是动态的，可能有多个子主题。


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to config file")
    parser.add_argument("--name", help="Name to be prepended to directory.")
    parser.add_argument("--notes", help="Additional information written to notes.txt inside the directory.")
    args = parser.parse_args()

    # config_path：必需的参数，指定配置文件的路径。
    #     --name：可选参数，指定日志文件夹的前缀名称。
    #     --notes：可选参数，允许用户附加一些说明文本，这些文本会被写入日志文件夹中的 notes.txt。

    # create the log directory
    stamp = datetime.datetime.now()
    ymd = stamp.strftime("%Y-%m-%d")
    hms = stamp.strftime("%H-%M-%S")
    if args.name is not None:
        dir_name = Path(ymd) / (args.name + "_" + hms)
    else:
        dir_name = Path(ymd) / hms

    log_dir = os.environ["MOBILE_MANIPULATION_CENTRAL_BAG_DIR"] / dir_name
    log_dir.mkdir(parents=True)

    # datetime.datetime.now()：获取当前时间。
    # 使用 strftime() 方法格式化当前时间为年月日（%Y-%m-%d）和时分秒（%H-%M-%S）。
    # 根据用户输入的 name，生成日志文件夹的名称。如果没有指定 name，则仅使用当前时间。
    # 日志文件夹将在指定的路径（通过环境变量 MOBILE_MANIPULATION_CENTRAL_BAG_DIR 设置）下创建。
    # 这个环境变量是可以修改的，直接在这个代码里更改就可以了

    # load configuration and write it out as a single yaml file
    '''
    config_out_path = log_dir / "config.yaml"
    config = core.parsing.load_config(args.config_path)
    with open(config_out_path, "w") as f:
        yaml.dump(config, stream=f, default_flow_style=False)
    '''
    # 从提供的配置路径加载 YAML 配置文件，使用 core.parsing.load_config() 来读取配置（这个函数在自定义的 upright_core 模块中）。
    # 将加载的配置数据写入日志文件夹中的 config.yaml 文件。
    #! 这个的作用：因为mpc需要很多的设置参数，所以这个是为了记录这些参数，方便后续的调试和分析。
    #! 在我们使用的时候，可以直接把那个info文件，复制过来，就可以把这步省略了。

    # write any notes
    if args.notes is not None:
        notes_out_path = log_dir / "notes.txt"
        with open(notes_out_path, "w") as f:
            f.write(args.notes)
    # 写入附加说明：如果用户提供了 notes 参数，将它写入日志文件夹中的 notes.txt 文件。

    # start the logging with rosbag
    rosbag_out_path = log_dir / "bag"
    rosbag_cmd = ROSBAG_CMD_ROOT + ["-o", rosbag_out_path] + ROSBAG_TOPICS
    proc = subprocess.Popen(rosbag_cmd)
    # 定义输出的 rosbag 路径（log_dir / "bag"）。
    # 构建完整的 rosbag 命令，包括输出路径和要记录的 ROS 主题。
    # 使用 subprocess.Popen() 启动 rosbag 命令，并在后台运行该命令以开始记录

    # spin until SIGINT (Ctrl-C) is received
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        # ctrl + c 退出


if __name__ == "__main__":
    main()

    # if __name__ == "__main__": 语句确保只有在脚本被直接执行时才会调用 main() 函数。
    # 如果该脚本作为模块被导入到其他地方，main() 函数就不会被执行。

'''
/clock：

这是 ROS 中的一个标准主题，通常用于记录时间戳信息。这对于同步不同机器人的数据和时间是非常重要的，尤其是在进行多机器人系统的调度或数据处理时。
--regex "/ridgeback/(.*)"：

记录所有以 /ridgeback/ 开头的主题。(.*) 表示匹配 /ridgeback/ 后的任意字符（包括子主题）。例如，这可以包含像 /ridgeback/sensors/imu 或 /ridgeback/control/cmd 等主题。具体内容取决于系统中如何设置和发布数据。
--regex "/ridgeback_velocity_controller/(.*)"：

记录所有以 /ridgeback_velocity_controller/ 开头的主题。这可能涉及到机器人 ridgeback 的速度控制数据，如 /ridgeback_velocity_controller/command 或 /ridgeback_velocity_controller/status。
--regex "/ur10/(.*)"：

记录所有以 /ur10/ 开头的主题。假设 ur10 是一个 UR10 机器人，那么相关主题可能包括 /ur10/pose（机器人位置）、/ur10/joint_states（关节状态）等。
--regex "/vicon/(.*)"：

记录所有以 /vicon/ 开头的主题。/vicon/ 可能指的是 Vicon motion capture 系统提供的数据，这些数据可以包括机器人的位置、姿态和运动数据。
--regex "/projectile/(.*)"：

记录所有以 /projectile/ 开头的主题。这个主题可能用于记录与飞行物体、弹道或其他类似项目相关的数据。具体取决于你的应用领域，这可以是 /projectile/position 或 /projectile/speed。
--regex "/mobile_manipulator_(.*)"：

记录所有以 /mobile_manipulator_ 开头的主题。假设系统中有一个移动机器人和机械臂结合的操作（比如 mobile_manipulator_1），那么这个主题可能包括 /mobile_manipulator_1/pose、/mobile_manipulator_1/status 等数据。
'''