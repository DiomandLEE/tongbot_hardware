"""General ROS parsing utilities."""
from pathlib import Path

import numpy as np
from spatialmath import UnitQuaternion
from spatialmath.base import qslerp
import xacro
import rospkg


UR10_JOINT_NAMES = [
    "ur10_arm_shoulder_pan_joint",
    "ur10_arm_shoulder_lift_joint",
    "ur10_arm_elbow_joint",
    "ur10_arm_wrist_1_joint",
    "ur10_arm_wrist_2_joint",
    "ur10_arm_wrist_3_joint",
]

KINOVA_JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7"
]

# maps joint names to the indices they are expected to have used to re-order
# feedback messages, which don't guarantee order
UR10_JOINT_INDEX_MAP = {name: index for index, name in enumerate(UR10_JOINT_NAMES)}

#反馈信息的顺序无法保证
KINOVA_JOINT_INDEX_MAP = {name: index for index, name in enumerate(KINOVA_JOINT_NAMES)}
# 得到的是 “name”： index，eg：‘joint_1’：1


def msg_time(msg):
    """Extract message timestamp as float in seconds."""
    return msg.header.stamp.to_sec()


def parse_time(msgs, normalize_time=True, t0=None):
    """Parse time in seconds from a list of messages.

    If normalize_time is True (the default), the array of time values will be
    normalized such that t[0] = t0. If t0 is not provided, it defaults to t[0].
    """
    t = np.array([msg_time(msg) for msg in msgs])
    if normalize_time:
        if t0:
            t -= t0
        else:
            t -= t[0]
    return t
    '''
    arse_time 函数的目的是从一组消息中提取时间戳，并根据需求将时间归一化。其输入为一个消息列表 msgs，
    并且可以选择是否对时间进行归一化。若 normalize_time 为 True（默认），
    则时间数组会以数组的第一个时间值或者给定的 t0 值为起点进行归一化，确保第一个时间戳为零。
    否则，返回的时间数组将保持原始顺序。具体过程包括：首先从每个消息中提取时间戳，
    然后根据 normalize_time 和是否提供 t0 来进行归一化处理，最后返回处理后的时间数组。
    '''

#for kinova gen3
def parse_kinova_joint_state_msg(msg):
    """Return a tuple (t, q, v) of time, configuration, velocity parsed from the
    JointState message of the Kinova Gen3.

    In particular, this correctly orders the joints.
    """
    t = msg_time(msg)

    # re-order joint names so the names correspond to indices given in
    # KINOVA_JOINT_INDEX_MAP
    q = np.zeros(7)
    v = np.zeros(7)
    for i in range(7):
        j = KINOVA_JOINT_INDEX_MAP[msg.name[i]]
        q[j] = msg.position[i]
        v[j] = msg.velocity[i]

    return t, q, v

def parse_kinova_joint_state_msgs(msgs, normalize_time=True):
    """Parse a list of Kinova Gen3 JointState messages.

    If normalize_time=True, the time array is shifted so that t[0] = 0."""
    ts = []
    qs = []
    vs = []

    for msg in msgs:
        t, q, v = parse_kinova_joint_state_msg(msg)
        ts.append(t)
        qs.append(q)
        vs.append(v)

    ts = np.array(ts)
    if normalize_time:
        ts -= ts[0]
    # np.array和list有着相同的地方，但是np.array提供了很多python的list列表没有的函数
    return ts, np.array(qs), np.array(vs)

# for ur10
def parse_ur10_joint_state_msg(msg):
    """Return a tuple (t, q, v) of time, configuration, velocity parsed from the
    JointState message of the UR10.

    In particular, this correctly orders the joints.
    """
    t = msg_time(msg)

    # re-order joint names so the names correspond to indices given in
    # UR10_JOINT_INDEX_MAP
    q = np.zeros(6)
    v = np.zeros(6)
    for i in range(len(msg.position)):
        j = UR10_JOINT_INDEX_MAP[msg.name[i]]
        q[j] = msg.position[i]
        v[j] = msg.velocity[i]

    return t, q, v


def parse_ur10_joint_state_msgs(msgs, normalize_time=True):
    """Parse a list of UR10 JointState messages.

    If normalize_time=True, the time array is shifted so that t[0] = 0."""
    ts = []
    qs = []
    vs = []

    for msg in msgs:
        t, q, v = parse_ur10_joint_state_msg(msg)
        ts.append(t)
        qs.append(q)
        vs.append(v)

    ts = np.array(ts)
    if normalize_time:
        ts -= ts[0]

    return ts, np.array(qs), np.array(vs)


def trim_msgs(msgs, t0=None, t1=None):
    """Trim messages that so only those in the time interval [t0, t1] are included."""
    ts = parse_time(msgs, normalize_time=False)
    start = 0
    if t0 is not None:
        for i in range(ts.shape[0]):
            if ts[i] >= t0:
                start = i
                break

    end = ts.shape[0]
    if t1 is not None:
        for i in range(start, ts.shape[0]):
            if ts[i] > t1:
                end = i
                break

    return msgs[start:end]
    '''
    `trim_msgs` 函数的作用是对一组消息进行时间裁剪，只保留时间区间 `[t0, t1]` 内的消息。如果消息的时间戳位于该时间区间外，它们将被丢弃。

### 代码解释：
1. **`parse_time(msgs, normalize_time=False)`**:
   - 这个函数调用会解析消息 `msgs` 中的时间戳，并返回一个时间数组 `ts`，时间戳不进行归一化（即直接提取消息中的原始时间）。

2. **初始化 `start` 和 `end`**:
   - `start` 变量用来确定消息裁剪的开始位置，默认为 0。
   - `end` 变量用来确定消息裁剪的结束位置，默认为消息列表的长度。

3. **裁剪开始时间 (`t0`)**:
   - 如果 `t0` 被指定（不为 `None`），那么遍历时间数组 `ts`，找到第一个时间戳大于等于 `t0` 的索引，将其作为 `start` 位置，从而确保开始时间在 `t0` 或之后。

4. **裁剪结束时间 (`t1`)**:
   - 如果 `t1` 被指定（不为 `None`），则从 `start` 位置开始，遍历时间数组，找到第一个时间戳大于 `t1` 的位置，将其作为 `end` 位置，确保结束时间在 `t1` 或之前。

5. **返回裁剪后的消息列表**：
   - 根据确定的 `start` 和 `end`，对输入的消息 `msgs` 进行切片，返回裁剪后的消息列表。

### 示例：
假设有以下时间戳的消息列表 `msgs`（时间戳：1.0, 2.0, 3.0, 4.0, 5.0）：
- 如果 `t0=2.0` 和 `t1=4.0`，则裁剪后会保留 `[2.0, 3.0, 4.0]` 时间段内的消息。
- 如果 `t0=None` 和 `t1=4.0`，则裁剪后会保留前四个消息，即时间戳 `[1.0, 2.0, 3.0, 4.0]`。

### 总结：
`trim_msgs` 函数的作用是根据给定的时间区间 `[t0, t1]` 来裁剪消息列表，确保返回的消息仅包含在该时间段内的数据。
    '''


def quaternion_from_msg(msg):
    """Parse a spatialmath quaternion from a geometry_msgs/Quaternion ROS message."""
    return UnitQuaternion(s=msg.w, v=[msg.x, msg.y, msg.z])


def yaw_from_quaternion_msg(msg):
    """Return the yaw component of a geometry_msgs/Quaternion ROS message."""
    Q = quaternion_from_msg(msg)
    return Q.rpy()[2]

def parse_dingo_vicon_msg(msg):
    """Get the base's (x, y, yaw) 2D pose from a geometry_msgs/TransformStamped ROS message from Vicon."""
    t = msg_time(msg)
    x = msg.transform.translation.x
    y = msg.transform.translation.y
    θ = yaw_from_quaternion_msg(msg.transform.rotation)
    q = np.array([x, y, θ])
    return t, q


def parse_dingo_vicon_msgs(msgs):
    """Parse a list of Vicon messages representing the base's pose."""
    ts = []
    qs = []
    for msg in msgs:
        t, q = parse_dingo_vicon_msg(msg)
        ts.append(t)
        qs.append(q)
    return np.array(ts), np.array(qs) # list变array，可使用的简单函数增多


def parse_dingo_joint_state_msgs(msgs, normalize_time=True):
    """Parse a list of Ridgeback JointState messages.

    If normalize_time=True, the time array is shifted so that t[0] = 0."""
    ts = []
    qs = []
    vs = []

    for msg in msgs:
        ts.append(msg_time(msg))
        qs.append(msg.position)
        vs.append(msg.velocity)

    ts = np.array(ts)
    if normalize_time:
        ts -= ts[0]

    return ts, np.array(qs), np.array(vs)

def parse_ridgeback_vicon_msg(msg):
    """Get the base's (x, y, yaw) 2D pose from a geometry_msgs/TransformStamped ROS message from Vicon."""
    t = msg_time(msg)
    x = msg.transform.translation.x
    y = msg.transform.translation.y
    θ = yaw_from_quaternion_msg(msg.transform.rotation)
    q = np.array([x, y, θ])
    return t, q


def parse_ridgeback_vicon_msgs(msgs):
    """Parse a list of Vicon messages representing the base's pose."""
    ts = []
    qs = []
    for msg in msgs:
        t, q = parse_ridgeback_vicon_msg(msg)
        ts.append(t)
        qs.append(q)
    return np.array(ts), np.array(qs)


def parse_ridgeback_joint_state_msgs(msgs, normalize_time=True):
    """Parse a list of Ridgeback JointState messages.

    If normalize_time=True, the time array is shifted so that t[0] = 0."""
    ts = []
    qs = []
    vs = []

    for msg in msgs:
        ts.append(msg_time(msg))
        qs.append(msg.position)
        vs.append(msg.velocity)

    ts = np.array(ts)
    if normalize_time:
        ts -= ts[0]

    return ts, np.array(qs), np.array(vs)


def parse_transform_stamped_msg(msg):
    """Parse time and pose from a TransformStamped message.

    Pose is represented as a length-7 vector with position followed by a
    quaternion representing orientation, with the scalar part of the quaternion
    at the end.
    """
    t = msg_time(msg)
    r = msg.transform.translation
    q = msg.transform.rotation
    pose = np.array([r.x, r.y, r.z, q.x, q.y, q.z, q.w])
    return t, pose


def parse_transform_stamped_msgs(msgs, normalize_time=True):
    """Parse a list of TransformStamped messages."""
    ts = parse_time(msgs, normalize_time=normalize_time)
    poses = np.array([parse_transform_stamped_msg(msg)[1] for msg in msgs])
    return ts, poses


def parse_wrench_stamped_msg(msg):
    """Parse time and wrench from a WrenchStamped message."""
    t = msg_time(msg)
    f = msg.wrench.force
    τ = msg.wrench.torque
    wrench = np.array([f.x, f.y, f.z, τ.x, τ.y, τ.z])
    return t, wrench


def parse_wrench_stamped_msgs(msgs, normalize_time=True):
    """Parse a list of WrenchStamped messages."""
    ts = parse_time(msgs, normalize_time=normalize_time)
    wrenches = np.array([parse_wrench_stamped_msg(msg)[1] for msg in msgs])
    return ts, wrenches


def lerp(x, y, s):
    """Linearly interpolate between values x and y with parameter s in [0, 1]."""
    assert 0 <= s <= 1
    return (1 - s) * x + s * y


def slerp(q0, q1, s):
    """Spherical linear interpolation between quaternions q0 and q1 with parameter s in [0, 1].

    Quaternions have order [x, y, z, w]; i.e., the scalar part comes at the end.
    """
    assert 0 <= s <= 1
    assert q0.shape == q1.shape == (4,)

    # we need to roll to convert between spatialmath's [w, x, y, z] convention
    # and our own
    q0 = np.roll(q0, 1)
    q1 = np.roll(q1, 1)
    q = qslerp(q0, q1, s)
    return np.roll(q, -1)


# TODO not ROS specific, so consider moving elsewhere
# 不是 ROS 特有的函数，可以考虑移到其他文件中，不适合ros_utils的命名
'''
自动处理时间点在原始时间范围之外的情况（填充首尾值）。对于超出old_time的时间点，直接复制首尾的值
'''
def interpolate_list(new_times, old_times, values, method="lerp"):
    """Align `values` (corresponding to `old_times`) with the `new_times` using
    interpolation.

    Each value in `values` should be a scalar or numpy array (i.e. something
    that can be scaled and added).

    `method` is the interpolation method. Linear interpolation `lerp` is the
    default. Alternative is `slerp` for spherical linear interpolation when the
    data to be interpolated is quaternions (in xyzw order).

    Returns a new list of values corresponding to `new_times`.
    """
    '''
    使用插值方法将 `values`（对应于 `old_times`）对齐到 `new_times`。
    `values` 中的每个值应为标量或 numpy 数组（即可以缩放和相加的类型）。
    `method` 是插值方法。默认是线性插值 `lerp`。如果插值的数据是四元数（按 xyzw 顺序），可以选择球形线性插值 `slerp`。
    返回与 `new_times` 对应的新值列表。

    功能：
        interpolate_list 函数接受四个参数：

        new_times: 新的时间序列，目标是对齐到这个时间序列。
        old_times: 原始时间序列，数据值 values 与其对应。
        values: 数据值，可以是标量或者 NumPy 数组，与 old_times 一一对应。
        method: 插值方法，默认是线性插值（lerp），另一个选项是球面线性插值（slerp，用于四元数数据）。
        返回值是一个新的值列表（aligned_values），这些值与 new_times 一一对应。
    '''
    if method == "slerp":
        interp_func = slerp
    else:
        interp_func = lerp
    # 根据 method 选择插值函数：
    #     lerp: 线性插值函数，用于一般标量或矢量数据。
    #     slerp: 球面线性插值函数，常用于四元数插值。

    aligned_values = []
    idx2 = 0
    n1 = len(new_times)
    n2 = len(old_times)
    # aligned_values 用来存储与 new_times 对齐的插值结果。
    # idx2 是一个索引变量，用于遍历 old_times。
    # n1 和 n2 分别是 new_times 和 old_times 的长度。

    for idx1 in range(n1): # 遍历 new_times：
        t = new_times[idx1]
        if idx1 > 0:
            assert t >= new_times[idx1 - 1], "Time moved backward!"

        # time is before the values start: pad with the first value
        if t <= old_times[0]:
            aligned_values.append(values[0])
            continue
        # 时间点早于原始数据的起始时间：
        #     if t <= old_times[0]:
        #         aligned_values.append(values[0])
        #         continue
        # 如果 t 小于或等于 old_times 的第一个时间戳，用第一个值 values[0] 填充。

        # time is after values end: pad with the last value
        if t >= old_times[-1]:
            aligned_values.append(values[-1])
            continue
        # 时间点晚于原始数据的起始时间：
        # 如果 t 大于或等于 old_times 的最后一个时间戳，用最后一个值 values[-1] 填充。

        # iterate through old_times until a more recent value found "遍历旧时间，直到找到一个更近的时间值。"
        while idx2 + 1 < n2 and old_times[idx2 + 1] < t: #n2是old_times的长度，且new_time中当前的t
            idx2 += 1
        # 使用循环更新 idx2，确保 t 位于 old_times[idx2] 和 old_times[idx2 + 1] 之间
        # 也就是说，在找寻old_time中距离t最近的idx

        assert old_times[idx2] <= t <= old_times[idx2 + 1]

        # interpolate between values[idx2] and values[idx2 + 1]
        Δt = old_times[idx2 + 1] - old_times[idx2]
        s = (t - old_times[idx2]) / Δt
        value = interp_func(values[idx2], values[idx2 + 1], s)
        #完成线性插值
        aligned_values.append(value)
        # 算时间间隔 Δt 和归一化比例 s。 调用插值函数 interp_func 在 values[idx2] 和 values[idx2 + 1] 之间进行插值。

    return aligned_values


def vicon_topic_name(name):
    return "/".join(["/vicon", name, name]) #得到的结果：/vicon/name/name


def package_path(package_name):
    """Get the path to a ROS package."""
    rospack = rospkg.RosPack()
    return Path(rospack.get_path(package_name))


def package_file_path(package_name, relative_path):
    """Get the path to a file within a ROS package.

    Parameters:
        package_name is the name of the ROS package
        relative_path is the path of the file relative to the package root

    Returns: Path object representing the file path.
    """
    return package_path(package_name) / relative_path
