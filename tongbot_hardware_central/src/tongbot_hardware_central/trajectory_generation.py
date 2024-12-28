import numpy as np

#todo 这个也要有推导的pdf
class PointToPointTrajectory:
    """A straight-line trajectory between two points.
    直线轨迹，从起点到终点。

    Parameters
    ----------
    start :
        The start point of the trajectory
        轨迹的起点
    delta :
        The position to travel to relative to start. In absolute terms, the
        goal point is ``start + delta``. We explicitly use ``delta`` here to
        allow the caller to handle e.g. wrapping values to π for angles.
        相对起点的位移，目标点是``start + delta``。我们显式使用``delta``来处理例如角度值的包裹问题。
        !说白了，delta就是目标点相对于起点的位移，这么做是为了防止处理角度值时出现的问题。比如，角度值在-π到π之间循环。
    timescaling :
        The timescaling to use.
        使用的时间缩放因子，比如对轨迹进行缩放，来调整速度和加速度，使得轨迹的持续时间满足速度和加速度约束。
    t0 : float
        Optionally provide the start time of the trajectory. If not provided,
        the first sample time will be taken as ``t0``.
        可选提供轨迹的起始时间。如果没有提供，第一次采样的时间将作为``t0``。
    """

    def __init__(self, start, delta, timescaling, t0=None):
        self.start = start  # 设置轨迹的起点
        self.delta = delta.reshape((delta.shape[0], 1))  # 目标点的相对位移，确保是列向量；shape[0]表示二维数组行数，一维数组的size
        self.goal = start + delta  # 计算目标点（起点 + 位移）
        self.timescaling = timescaling  # 设置时间缩放
        self.t0 = t0  # 设置起始时间

    @classmethod  #类方法，不需要实例化就可以调用
    def quintic(cls, start, delta, max_vel, max_acc, t0=None, min_duration=None):
        """Construct the trajectory using a quintic timescaling with duration
        suitable to satisfy velocity and acceleration constraints.
        使用五次时间缩放构造轨迹，持续时间适合满足速度和加速度约束。

        Parameters
        ----------
        start :
            The start point of the trajectory
            轨迹的起点
        delta :
            The position to travel to relative to start. In absolute terms, the
            goal point is ``start + delta``.
            相对起点的位移，目标点是``start + delta``。
        max_vel : float
            Maximum allowable velocity.
            最大允许速度
        max_acc : float
            Maximum allowable acceleration.
            最大允许加速度
        t0 : float
            Optionally provide the start time of the trajectory. If not provided,
            the first sample time will be taken as ``t0``.
            可选提供轨迹的起始时间。如果没有提供，第一次采样的时间将作为``t0``。
        min_duration : float
            Optionally provide a minimum duration for the trajectory. If the
            timescaling obeying the maximum velocity and acceleration limits
            has a lower duration, we replace the timescaling with one of
            ``min_duration``.
            可选提供轨迹的最小持续时间。如果使用最大速度和加速度限制的时间缩放计算出的持续时间较短，我们将用``min_duration``替换时间缩放。
        """
        distance = np.max(np.abs(delta))  # 计算最大位移距离
        timescaling = QuinticTimeScaling.from_max_vel_acc(distance, max_vel, max_acc)  # 根据最大速度和加速度来计算时间缩放；是一个类方法属性，可以直接调用类名来调用
        if min_duration is not None and timescaling.duration < min_duration:
            timescaling = QuinticTimeScaling(min_duration)  # 如果计算出的时间小于最小持续时间，则使用最小持续时间
        return cls(start, delta, timescaling, t0)  # 返回创建的轨迹对象

    @property
    def duration(self):
        """The duration of the trajectory.
        轨迹的持续时间"""
        return self.timescaling.duration  # 返回轨迹的持续时间

    @property
    def max_velocity(self):
        """The maximum velocity of the trajectory.
        轨迹的最大速度

        This value is always positive, even if the true velocity is negative.
        即使实际速度为负，该值也始终为正。
        """
        return np.abs(np.squeeze(self.timescaling.max_ds * self.delta))  # 返回最大速度，确保返回正值

    @property
    def max_acceleration(self):
        """The maximum acceleration of the trajectory.
        轨迹的最大加速度

        This value is always positive, even if the true acceleration is
        negative.
        即使实际加速度为负，该值也始终为正。
        """
        #! 去除维度为1的维度，若是1*1则返回一个标量
        return np.abs(np.squeeze(self.timescaling.max_dds * self.delta))  # 返回最大加速度，确保返回正值

    def done(self, t):
        """Check if the trajectory is done at the given time.
        检查轨迹是否在给定时间完成

        Parameters
        ----------
        t : float
            Time.
            时间

        Returns
        -------
        :
            ``True`` if the trajectory is done, ``False`` otherwise. Also
            returns ``False`` if no initial time ``t0`` was supplied and the
            trajectory has not yet been sampled.
            如果轨迹完成，则返回``True``, 否则返回``False``。如果没有提供起始时间``t0``且轨迹尚未采样，则也返回``False``。
        """
        #! 简单地检查当前时间是否已经超过轨迹的时间区间
        if self.t0 is None:
            return False  # 如果没有提供起始时间，则默认还没有开始
        return t - self.t0 >= self.duration  # 如果当前时间已经超过轨迹的持续时间，则表示轨迹完成

    def sample(self, t):
        """Sample the trajectory at the given time.
        在给定时间采样轨迹

        If ``t0`` was not supplied when the trajectory was constructed, then
        ``t0`` will be set to the first ``t`` at which it is sampled.
        如果在轨迹构造时未提供``t0``，则``t0``将设置为第一次采样的时间。

        If ``t`` is beyond the end of the trajectory, then the final point of
        the trajectory is returned; i.e., all calls to ``self.sample`` with
        ``t >= self.duration`` return the same values.
        如果``t``超出了轨迹的结束时间，则返回轨迹的最后一点；即，所有``self.sample``的调用，如果``t >= self.duration``，都会返回相同的值。

        Parameters
        ----------
        t : float
            The sample time.
            采样时间

        Returns
        -------
        :
            A tuple (position, velocity, acceleration) for the sample time.
            返回一个元组，包含采样时间下的位置、速度和加速度。
        """
        if self.t0 is None:
            self.t0 = t  # 如果没有提供起始时间，则将第一次采样的时间（就是把输入的t作为起始时间）作为起始时间

        if self.done(t):
            return self.goal, np.zeros_like(self.goal), np.zeros_like(self.goal)  # 如果轨迹已经完成，返回目标点和零速度、加速度

        #! 使用t-self.t0来计算时间变量的值
        s, ds, dds = self.timescaling.eval(t - self.t0)  # 使用时间缩放来计算轨迹在当前时间的插值
        #! 因为之前在计算系数coeffs时，是将系数都除以了总路程长度，所以这里在计算s,ds,dds的时候，需要乘回来
        position = self.start + (s * self.delta).T  # 计算当前位置
        velocity = (ds * self.delta).T  # 计算速度
        acceleration = (dds * self.delta).T  # 计算加速度
        if np.ndim(t) == 0: #判断时间是否为标量；可能是因为t是一个维度是自由度个数的列向量
            position = np.squeeze(position)  # 如果时间是标量，则将结果降维
            velocity = np.squeeze(velocity)
            acceleration = np.squeeze(acceleration)
        return position, velocity, acceleration  # 返回当前位置、速度和加速度


class QuinticTimeScaling:
    """Quintic time-scaling with zero velocity and acceleration at end points.
    五次时间缩放，端点处速度和加速度为零。

    Parameters
    ----------
    duration : float
        Non-negative duration of the trajectory.
        轨迹的非负持续时间

    Attributes
    ----------
    coeffs :
        The coefficients of the time-scaling equation.
        时间缩放方程的系数
    max_ds : float
        The maximum value of the first time derivative.
        第一时间导数的最大值
    max_dds : float
        The maximum value of the second time derivative.
        第二时间导数的最大值

    Raises
    ------
    AssertionError
        If the ``duration`` is negative.
        如果``duration``为负，则抛出异常。
    """

    def __init__(self, duration):
        assert duration >= 0  # 确保持续时间是非负的
        self.duration = duration  # 保存持续时间； 轨迹的用时
        T = duration  # 将持续时间赋值给T
        A = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [1, T, T**2, T**3, T**4, T**5],
                [0, 1, 0, 0, 0, 0],
                [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
                [0, 0, 2, 0, 0, 0],
                [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
            ]
        )  # 创建方程的系数矩阵
        b = np.array([0, 1, 0, 0, 0, 0])  # 右边的常数向量；a0 = a0 / L,L是总的距离长度
        #! 对应的是五次多项式，在t=0时，s=0，在t=T时，s=1，在t=0时，ds=0，在t=T时，ds=0，在t=0时，dds=0，在t=T时，dds=0
        self.coeffs = np.linalg.solve(A, b)  # 解线性方程组，得到系数

        #! https://blog.csdn.net/maple_2014/article/details/104717333 五次多项式讲解
        self.max_ds = 15 / (8 * T)  # 计算最大速度的系数
        self.max_dds = 10 / (T**2 * np.sqrt(3))  # 计算最大加速度的系数

    @classmethod
    def from_max_vel_acc(cls, distance, max_vel, max_acc):
        """Make a timescaling with long enough duration to satisfy bounds on
        velocity and acceleration.
        根据最大速度和加速度要求，计算合适的时间缩放。

        Parameters
        ----------
        distance : float
            The maximum distance to be travelled.
            要旅行的最大距离
        max_vel : float
            The maximum allowable velocity.
            最大允许速度
        max_acc : float
            The maximum allowable acceleration.
            最大允许加速度

        Raises
        ------
        AssertionError
            If ``distance``, ``max_vel``, or ``max_acc`` are negative.
            如果``distance``、``max_vel``或``max_acc``为负，则抛出异常。
        """
        assert distance >= 0  # 确保距离是非负的
        assert max_vel >= 0  # 确保最大速度是非负的
        assert max_acc >= 0  # 确保最大加速度是非负的

        max_ds = max_vel / distance  # 计算最大速度的时间衰减系数
        max_dds = max_acc / distance  # 计算最大加速度的时间衰减系数
        #! 这里的参数选择是为了松弛这些时间，选择一个总时长的最大值，其实就是一个以vm为平均速度，以8/15松弛，另一个以am为平均加速度，以√3/5松弛，取最大值
        T = max(15 / (max_ds * 8), np.sqrt(10 / (np.sqrt(3) * max_dds)))  # 计算满足最大速度和加速度约束的最小时间
        return cls(T)  # 返回时间缩放对象

    def eval(self, t):
        """Evaluate the timescaling at a given time.
        在给定时间评估时间缩放

        Parameters
        ----------
        t : float
            Time, must be in the interval ``[0, self.duration]``.
            时间，必须在``[0, self.duration]``区间内

        Returns
        -------
        :
            A tuple (s, ds, dds) representing the value of the time-scaling
            and its first two derivatives at the time ``t``.
            返回一个元组（s, ds, dds），表示在时间``t``时，时间缩放及其前两阶导数的值。

        Raises
        ------
        AssertionError
            If ``t`` is not in the interval ``[0, self.duration]``.
            如果``t``不在``[0, self.duration]``区间内，抛出异常。
        """
        #! 就是利用多项式的导数求速度和加速度
        assert 0 <= t <= self.duration  # 确保时间在有效区间内
        #! np.ones_like(t)用于创建一个与给定数组 t 形状相同，但所有元素都为 1 的数组。
        #! .dot():用于计算两个数组的点积。对于一维数组，点积就是对应元素的乘积之和。对于二维数组（矩阵），点积是矩阵乘法。
        s = self.coeffs.dot([np.ones_like(t), t, t**2, t**3, t**4, t**5])  # 计算时间缩放的值
        ds = self.coeffs[1:].dot(
            [np.ones_like(t), 2 * t, 3 * t**2, 4 * t**3, 5 * t**4]
        )  # 计算速度
        dds = self.coeffs[2:].dot(
            [2 * np.ones_like(t), 6 * t, 12 * t**2, 20 * t**3]
        )  # 计算加速度
        return s, ds, dds  # 返回时间、速度和加速度
