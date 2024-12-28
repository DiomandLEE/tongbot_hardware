import datetime
import os
from pathlib import Path
import subprocess
import signal

import rospy
from geometry_msgs.msg import TransformStamped

from tongbot_hardware_central.ros_utils import vicon_topic_name

'''
导入必要的模块：
- datetime: 用于获取当前时间。
- os: 用于操作系统功能，例如读取环境变量。
- pathlib.Path: 用于路径操作。
- subprocess: 用于启动外部进程。
- signal: 用于发送信号来控制进程（如中断信号）。
- rospy: ROS Python 客户端库。
- TransformStamped: 用于处理 Vicon 数据消息类型的 ROS 消息类型。
- vicon_topic_name: 自定义的函数，生成 Vicon 话题名称。
'''


BAG_DIR_ENV_VAR = "tongbot_hardware_BAG_DIR"
BAG_DIR = os.environ.get(BAG_DIR_ENV_VAR, None)
# 从环境变量中获取日志目录，如果未设置则设为 None。应该是要在bashrc里面设置这个环境变量

ROSBAG_CMD_ROOT = ["rosbag", "record"]


class DataRecorder:
    def __init__(self, topics, name=None, root=None, notes=None):
        if root is None:  # 设置录包存储的根目录
            if BAG_DIR is None:
                raise ValueError(
                    f"No root directory given and {BAG_DIR_ENV_VAR} environment variable not set."
                )
            root = BAG_DIR
        '''
        构造函数：初始化 `DataRecorder` 类的对象。
        - `topics`: 需要记录的 ROS 话题。
        - `name`: 保存数据的文件夹名。
        - `root`: 存储根目录，如果没有提供，则使用环境变量 `BAG_DIR`。
        如果根目录 `root` 没有提供，且环境变量 `BAG_DIR` 为空，则抛出异常。
        '''

        stamp = datetime.datetime.now()
        ymd = stamp.strftime("%Y-%m-%d")
        hms = stamp.strftime("%H-%M-%S")
        if name is not None:
            dir_name = Path(ymd) / (name + "_" + hms)
        else:
            dir_name = Path(ymd) / hms
        '''
        获取当前时间，并生成文件夹名称：
        - `ymd`: 当前日期，格式为 `年-月-日`。
        - `hms`: 当前时间，格式为 `小时-分钟-秒`。
        如果提供了 `name`，则文件夹名将包含该名称和当前时间；否则，只使用时间戳。
        '''

        self.log_dir = root / dir_name
        self.topics = topics
        self.notes = notes
        '''
        设置日志目录：
        - `log_dir`: 最终的保存路径。
        - `topics`: 需要记录的 ROS 话题。
        - `notes`: 记录的附加注释。
        '''

    '''
    Python中，函数名前加下划线（_）通常表示这是一个内部函数或方法，不应该在模块外部直接调用。
    这是一种约定俗成的命名方式，用于向其他开发者表明这个函数或方法是私有的
    '''

    def _mkdir(self):
        self.log_dir.mkdir(parents=True)  # 创建日志目录，如果父目录不存在则创建。

    def _record_notes(self):
        if self.notes is not None:
            notes_out_path = self.log_dir / "notes.txt"
            with open(notes_out_path, "w") as f:
                f.write(self.notes)  # 将注释写入文件 "notes.txt"
                #! 就是给了一个注释，并把注释写入文件，/home/user/roslogs/2024-12-11/12-34-56/notes.txt

    def _record_bag(self):
        rosbag_out_path = self.log_dir / "bag"
        rosbag_cmd = ROSBAG_CMD_ROOT + ["-o", rosbag_out_path] + self.topics
        self.proc = subprocess.Popen(rosbag_cmd)  # 启动 rosbag 进程来记录数据
        #! 最后的bag文件路径： for example：/home/user/roslogs/2024-12-11/12-34-56/bag.bag
    #! 相当于发布这个命令：
    #!   rosbag record -o /home/user/roslogs/2024-12-11/12-34-56/bag /topic1 /topic2
    #!  -o：是 rosbag 命令的选项，用于指定输出文件的基本名称。

    def record(self):
        self._mkdir()  # 创建日志目录
        self._record_notes()  # 记录注释
        self._record_bag()  # 启动 rosbag 录制

    def close(self):
        self.proc.send_signal(signal.SIGINT)  # 发送 SIGINT 信号中断 rosbag 进程，停止录制


class ViconRateChecker:
    def __init__(self, vicon_object_name, duration=5):
        assert duration > 0  # 确保持续时间大于0
        self.duration = duration  # 持续时间（秒）
        self.msg_count = 0  # 用于统计接收到的消息数量
        self.start_time = None  # 记录开始时间
        self.started = False  # 标志位，表示是否开始计时
        self.done = False  # 标志位，表示检查是否完成
        self.topic_name = vicon_topic_name(
            vicon_object_name)  # 获取 Vicon 对象对应的topic名称
        self.vicon_sub = rospy.Subscriber(
            self.topic_name, TransformStamped, self._vicon_cb
        )  # 订阅 Vicon 数据话题

    def _vicon_cb(self, msg):
        if not self.started:
            return  # 如果未开始计时，则直接返回

        now = rospy.Time.now().to_sec()  # 获取当前时间
        if self.start_time is None:
            self.start_time = now  # 如果没有开始时间，初始化为当前时间

        if now - self.start_time > self.duration:
            self.vicon_sub.unregister()  # 取消订阅
            self.done = True  # 设置检查完成标志
            return

        self.msg_count += 1  # 增加接收到的消息数量

    def check_rate(self, expected_rate, bound=1, verbose=True):
        self.started = True  # 设置为开始计时

        rate = rospy.Rate(1)  # 每秒检查一次
        while not self.done and not rospy.is_shutdown():  # 没检查完且ros还在运行
            rate.sleep()  # 每次循环休眠直到下一次
            if self.msg_count == 0:
                # 如果没有接收到任何消息，输出提示
                print(f"I haven't received any messages on {self.topic_name}")

        rate = self.msg_count / self.duration  # 计算实际的接收频率；
        # 因为回调函数中，定义了duration，hz就是每秒跑多少次，所以除以duration，得到的就是频率hz
        lower = expected_rate - bound  # 计算允许的最小频率
        upper = expected_rate + bound  # 计算允许的最大频率

        if verbose:
            print(
                f"Received {self.msg_count} Vicon messages over {self.duration} seconds."
            )  # 输出接收到的消息数量
            print(f"Expected Vicon rate = {expected_rate} Hz")  # 输出期望的频率
            print(f"Average Vicon rate = {rate} Hz")  # 输出实际接收到的频率

        return lower <= rate <= upper  # 返回是否在允许范围内
#! 目前来看，这个ViconRateChecker类并没有用到，需要的时候可以参考record.py来书写
