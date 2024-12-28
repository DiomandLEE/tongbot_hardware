import time
import signal
import numpy as np
import rospy
from spatialmath.base import rotz
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tongbot_hardware_central import ros_utils

# for kinova
import sys
import os
import time
import threading

#! 这一部分是安装在系统里
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient

from kortex_api.autogen.messages import Session_pb2, Base_pb2, Common_pb2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utilities

# Vicon物体接口类：用于接收Vicon系统的物体位姿数据


class ViconObjectInterface:
    """ROS interface for receiving Vicon measurements for an object's pose."""

    def __init__(self, name):
        """
        初始化Vicon物体接口，通过ROS订阅Vicon发布的物体位置和姿态。
        参数:
        name: 物体在Vicon系统中的名称，用于生成订阅主题。
        """
        topic = "/vicon/" + name + "/" + name  # 根据物体名称构建订阅主题
        self.msg_received = False  # 标记是否收到消息
        self.sub = rospy.Subscriber(
            topic, TransformStamped, self._transform_cb)  # 订阅Vicon话题

    def ready(self):
        """如果收到Vicon消息，返回True。"""
        return self.msg_received

    def _transform_cb(self, msg):
        """
        Vicon数据回调函数，将收到的物体位姿数据转换成numpy数组。
        参数:
        msg: Vicon发布的TransformStamped消息，包含物体的位置和旋转。
        """
        L = msg.transform.translation  # 获取位置
        Q = msg.transform.rotation  # 获取四元数表示的旋转
        self.position = np.array([L.x, L.y, L.z])  # 将位置转换为numpy数组
        self.orientation = np.array([Q.x, Q.y, Q.z, Q.w])  # 将旋转转换为四元数
        self.msg_received = True  # 设置消息已收到标志

# 机器人ROS接口基类：提供基本的ROS接口功能


class RobotROSInterface:
    """Base class for defining ROS interfaces for robots."""

    def __init__(self, nq, nv):
        """
        初始化机器人ROS接口。
        参数:
        nq: 机器人关节数
        nv: 机器人速度维度
        """
        self.nq = nq  # 关节数
        self.nv = nv  # 速度维度
        self.q = np.zeros(self.nq)  # 初始化关节位置为零
        self.v = np.zeros(self.nv)  # 初始化关节速度为零
        self.joint_states_received = False  # 标记是否接收到关节状态

    def brake(self):
        """刹车（停止）机器人，通过发布零速度命令。"""
        self.publish_cmd_vel(np.zeros(self.nv))

    def ready(self):
        """如果接收到关节状态消息，返回True。"""
        return self.joint_states_received

# Dingo移动机器人ROS接口类


class DingoROSInterface(RobotROSInterface):
    """ROS interface for the Dingo mobile base."""

    def __init__(self):
        """初始化DingoROS接口，设置关节数和速度维度，并订阅关节状态话题。"""
        super().__init__(nq=3, nv=3)
        self.cmd_pub = rospy.Publisher(
            "/ridgeback/cmd_vel", Twist, queue_size=1)  # 发布速度命令
        self.joint_state_sub = rospy.Subscriber(
            "/ridgeback/joint_states", JointState, self._joint_state_cb  # 订阅关节状态话题
        )

    def _joint_state_cb(self, msg):
        """Dingo关节状态反馈回调函数，将位置和速度更新到实例变量中。"""
        self.q = np.array(msg.position)
        self.v = np.array(msg.velocity)
        self.joint_states_received = True  # 标记关节状态已收到

    def publish_cmd_vel(self, cmd_vel, bodyframe=False):
        """
        发布速度命令。
        参数:
        cmd_vel: 速度命令，包含线速度和角速度
        bodyframe: 如果为True，表示速度命令是在底盘坐标系下，默认是世界坐标系
        """
        assert cmd_vel.shape == (self.nv,)  # 确保命令的维度正确
        if not bodyframe:
            C_bw = rotz(-self.q[2])  # 将速度命令从世界坐标系转换到底盘坐标系
            cmd_vel = C_bw @ cmd_vel  # 矩阵乘法进行坐标转换
        msg = Twist()
        msg.linear.x = cmd_vel[0]  # 设置线速度
        msg.linear.y = cmd_vel[1]
        msg.angular.z = cmd_vel[2]  # 设置角速度
        self.cmd_pub.publish(msg)  # 发布速度命令

# Kinova机械臂ROS接口类


class KinovaROSInterface(RobotROSInterface):
    """ROS interface for the Kinova arm."""

    def __init__(self):
        """初始化KinovaROS接口，设置关节数和速度维度，并订阅关节状态话题。"""
        super().__init__(nq=7, nv=7)
        self.joint_state_sub = rospy.Subscriber(
            "/my_gen3/joint_states", JointState, self._joint_state_cb  # 订阅Kinova关节状态话题
        )
        self.cmd_pub = rospy.Publisher(
            "/my_gen3/cmd_vel", Float64MultiArray, queue_size=1) # for ros record bag
        args = utilities.parseConnectionArguments()
        router = utilities.DeviceConnection.createTcpConnection(args)
        self.base = BaseClient(router)  # 创建Kinova机械臂客户端
        # self.kinova_notify(self.base)

    def _joint_state_cb(self, msg):
        """Kinova关节状态回调函数，解析并更新关节位置和速度。"""
        _, self.q, self.v = ros_utils.parse_kinova_joint_state_msg(
            msg)  # 使用工具函数解析关节状态消息，todo，需要在utils中添加，注意是添加不是修改
        self.joint_states_received = True  # 标记关节状态已收到

    def publish_cmd_vel(self, cmd_vel, bodyframe=None):
        """发布Kinova机械臂的关节速度命令。"""
        assert cmd_vel.shape == (self.nv,)

        joint_speeds = Base_pb2.JointSpeeds()
        speeds = list(cmd_vel)

        i = 0
        for speed in speeds:
            joint_speed = joint_speeds.joint_speeds.add()
            joint_speed.joint_identifier = i
            joint_speed.value = speed
            joint_speed.duration = 0
            i = i + 1
        self.base.SendJointSpeedsCommand(joint_speeds)

        msg = Float64MultiArray()
        msg.data = list(cmd_vel)
        self.cmd_pub.publish(msg) # for ros bag，这个还要在C++里进行完善

    def check_for_end_or_abort(e):
        """Return a closure checking for END or ABORT notifications

        Arguments:
        e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
        """
        def check(notification, e = e):
            print("EVENT : " + \
                Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END \
            or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()
        return check

    def kinova_notify(self):
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
        self.check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
        )

        # finished = e.wait(0.1)
        self.base.Unsubscribe(notification_handle)
# 移动操控机器人ROS接口类，结合了底盘和机械臂接口


class MobileManipulatorROSInterface:
    """ROS interface to the real mobile manipulator."""

    def __init__(self):
        """初始化移动操控机器人接口，包括Kinova机械臂和Dingo底盘。"""
        self.arm = KinovaROSInterface()  # 初始化Kinova Gen3机械臂接口
        self.base = DingoROSInterface()  # 初始化Dingo底盘接口
        self.nq = self.arm.nq + self.base.nq  # 计算总关节数
        self.nv = self.arm.nv + self.base.nv  # 计算总速度维度

    def brake(self):
        """刹车（停止）机器人，停止底盘和机械臂。"""
        self.base.brake()
        self.arm.brake()

    def ready(self):
        """如果底盘和机械臂的关节状态都已接收到，返回True。"""
        return self.base.ready() and self.arm.ready()

    def publish_cmd_vel(self, cmd_vel, bodyframe=False):
        """发布机器人整体的速度命令，分别传递给底盘和机械臂。"""
        assert cmd_vel.shape == (self.nv,)
        self.base.publish_cmd_vel(
            cmd_vel[: self.base.nv], bodyframe=bodyframe)  # 发布底盘的速度命令,#! :表示从首到base.nv
        self.arm.publish_cmd_vel(cmd_vel[self.base.nv:])  # 发布机械臂的速度命令 #! :表示从base.nv

    @property    # 装饰器，将方法转换为属性，可以直接通过对象访问
    def q(self):
        """返回最新的关节配置（包括底盘和机械臂的关节位置）。"""
        return np.concatenate((self.base.q, self.arm.q))  # 将底盘和机械臂的关节位置合并

    @property
    def v(self):
        """返回最新的关节速度（包括底盘和机械臂的速度）。"""
        return np.concatenate((self.base.v, self.arm.v))

# 机器人信号处理类：用于在收到退出信号时刹车机器人


class RobotSignalHandler:
    """Custom signal handler to brake the robot before shutting down ROS."""

    def __init__(self, robot, dry_run=False):
        """初始化信号处理器，设置机器人和是否干运行模式。"""
        '''dry_run 是一个常用的参数，通常用于模拟执行某个操作而不真正执行该操作。
        通过设置 dry_run=True，程序可以在实际执行任务之前先展示将要执行的步骤或行为，
        确保没有意外或不必要的副作用。这个参数的目的是让用户验证操作是否按预期进行，而不产生实际的影响。'''
        self.robot = robot
        self.dry_run = dry_run
        signal.signal(signal.SIGINT, self.handler)  # 处理SIGINT信号（通常为Ctrl+C）
        signal.signal(signal.SIGTERM, self.handler)  # 处理SIGTERM信号（程序终止）

    def handler(self, signum, frame):
        """收到退出信号时的处理函数，刹车机器人并安全关闭ROS。"""
        print("Received SIGINT.")
        if not self.dry_run:
            print("Braking robot.")
            self.robot.brake()  # 刹车机器人
            time.sleep(0.1)  # 延时，确保刹车有效
        rospy.signal_shutdown("Caught SIGINT!")  # 关闭ROS

        # for C++
        # void signalHandler(int signum)
        # {
        #     ROS_INFO("Caught signal %d, shutting down ROS.", signum);
        #     ros::shutdown();  // 关闭 ROS
        # }

        # 当程序接收到 SIGINT 或 SIGTERM 信号时，你可以执行一些清理操作，比如保存数据、关闭文件、停止硬件操作等。这通常比程序直接崩溃或强制终止更安全。

# 简单信号处理类：用于简单地捕获信号并设置标志


class SimpleSignalHandler:
    """Simple signal handler that just sets a flag when a signal has been caught.
    简单的信号处理程序，当捕获到信号时只设置一个标志"""

    def __init__(self, sigint=True, sigterm=False, callback=None):
        """初始化信号处理器，设置捕获的信号类型和回调函数。"""
        self.received = False
        self._callback = callback
        if sigint:
            signal.signal(signal.SIGINT, self.handler)  # 捕获SIGINT信号
        if sigterm:
            signal.signal(signal.SIGTERM, self.handler)  # 捕获SIGTERM信号
            #! 除了使用ctrl+c中断外，
            #! 其余的(rosnode kill、开启其他的launch文件、代码中的rospy.signal_shutdown()，ros::shutdown()会触发SIGTERM信号

    def handler(self, signum, frame):
        """处理捕获到的信号，设置received标志并调用回调函数。"""
        print(f"Received signal: {signum}")
        self.received = True
        if self._callback is not None:
            self._callback()  # 调用回调函数（如果有的话）
