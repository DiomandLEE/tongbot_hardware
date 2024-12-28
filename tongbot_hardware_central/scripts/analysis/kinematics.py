import numpy as np
import tongbot_hardware_central as tongbot
import IPython

'''
这个是用来测试kinematics的
'''
HOME_NAME = "projectile2"


def main():
    home = tongbot.load_home_position(HOME_NAME)
    kinematics = tongbot.MobileManipulatorKinematics()

    J = kinematics.jacobian(home)
    U, S, VT = np.linalg.svd(J[:3, :])

    # the direction which we can instantaneously move fastest in joint space
    # (ignoring constraints) corresponds to the largest singular value of J (or
    # largest eigenvalue of JJ^T)
    '''
    我们可以在关节空间中瞬时移动最快的的方向
    （忽略约束条件）对应于雅可比矩阵 J 的最大奇异值（或 JJ^T 的最大特征值）
    '''
    n = U[:, 0]
    dqdt = VT[0, :]

    '''
    n = U[:, 0]：这是雅可比矩阵的第一列（即U 的第一列），对应的是任务空间中最快的瞬时运动方向。
    在关节空间中，它表示机器人关节如何变化以最快地朝着任务空间的某个方向运动。数学上，它对应于最大的奇异值方向。

    dqdt = VT[0, :]：这是雅可比矩阵右奇异向量V^T的第一行，
    它对应于关节空间中与最快运动方向相关的速度分量。也就是说，这是在任务空间最快的运动方向下，各个关节的变化速度。
    '''

    IPython.embed()


if __name__ == "__main__":
    main()
