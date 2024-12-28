#pragma once

#include <geometry_msgs/TransformStamped.h> // 包含ROS消息类型，用于接收Vicon数据
#include <mobile_manipulation_central/kalman_filter.h> // 自定义的卡尔曼滤波器头文件
#include <ros/console.h> // ROS日志系统
#include <ros/ros.h> // ROS核心库
#include <sensor_msgs/JointState.h> // 包含ROS消息类型，用于发布关节状态
#include <std_msgs/Empty.h> // 包含空消息类型，用于重置估计器
#include <tf/transform_datatypes.h> // 包含TF变换相关的工具函数

#include <Eigen/Eigen> // 包含Eigen库，用于矩阵运算

namespace tongbot { // 定义tongbot命名空间

// The node listens to raw Vicon transform messages for a projectile object and
// converts them to a JointState message for the translational component. For
// the moment we assume we don't care about the projectile orientation and so
// do not do anything with it.
// 该节点监听来自Vicon系统的原始变换消息，用于一个弹道物体，并将其转换为关节状态消息的平移分量。
// 目前我们假设不需要关心物体的方向，因此不对其进行任何处理。
class ProjectileViconEstimator { // 定义ProjectileViconEstimator类
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW //! 确保Eigen对象正确对齐
    /*
    具体来说，EIGEN_MAKE_ALIGNED_OPERATOR_NEW 宏会重载类的 operator new 和 operator delete，
    以确保通过 new 分配的对象满足 Eigen 所需的内存对齐要求。这对于提高性能至关重要，特别是在处理大型矩阵和向量时。
    */

    ProjectileViconEstimator() {} // 默认构造函数

    bool init(ros::NodeHandle& nh, double proc_var, double meas_var,
              double nis_bound, double activation_height,
              double deactivation_height, const Eigen::Vector3d& gravity) {
        proc_var_ = proc_var; // 过程噪声方差，状态转移过程，即运动过程
        meas_var_ = meas_var; // 测量噪声方差
        nis_bound_ = nis_bound; // NIS边界值
        activation_height_ = activation_height; // 激活高度
        deactivation_height_ = deactivation_height; // 去激活高度
        gravity_ = gravity; // 重力加速度
        return true;
    }

    // True if enough messages have been received so all data is initialized.
    // 如果接收到足够的消息以初始化数据，则返回true
    bool ready() const { return msg_count_ > 0; } //? 累计一定的vicon测量值，才ready

    // Get number of Vicon messages received
    // 获取接收到的Vicon消息数量
    size_t get_num_msgs() const { return msg_count_; } //接收的vicon消息数量

    // Get most recent position
    // 获取最近的位置估计
    Eigen::Vector3d q() const { return estimate_.x.head(3); }

    // Get most recent velocity
    // 获取最近的速度估计
    Eigen::Vector3d v() const { return estimate_.x.tail(3); }
    //! 从上面来看，状态就是位置和速度，3，xyz

    // Reset the estimator (estimate goes back to initial conditions)
    // 重置估计器（估计回到初始条件）
    void reset() {
        msg_count_ = 0; // 重置消息计数
        active_ = false; // 设置为非激活状态
    }

    // Update the estimator at time t with measurement y of the projectile's
    // position.
    // 在时间t,用弹道物体的位置测量值y,更新估计器
    void update(const double t, const Eigen::Vector3d& y) {
        // We assume the projectile is in flight (i.e. subject to gravitational
        // acceleration) if it is above a certain height and has not yet
        // reached a certain minimum height. Otherwise, we assume acceleration
        // is zero.
        // 我们假设如果弹道物体的高度超过某个阈值且尚未达到最小高度，则认为它在飞行中（即受重力加速度影响）。
        // 否则，我们认为加速度为零。
        // TODO probably better to use the estimate for this
        if (active_ && y(2) <= deactivation_height_) {
            active_ = false; // 如果高度低于去激活阈值，则设置为非激活状态
        }
        if (!active_ && y(2) >= activation_height_) {
            active_ = true; // 如果高度高于激活阈值，则设置为激活状态
        }

        if (msg_count_ == 0) {
            // Initialize the estimate
            // 初始化估计
            estimate_.x = Eigen::VectorXd::Zero(6); // 初始化状态向量 0？
            estimate_.P = Eigen::MatrixXd::Identity(6, 6);  // 初始化协方差矩阵 协方差是1？
        } else if (msg_count_ >= 1) {
            double dt = t - t_prev_; // 计算时间步长

            // Compute system matrices
            // 计算系统矩阵
            Eigen::Matrix3d I = Eigen::Matrix3d::Identity(); // 单位矩阵

            Eigen::MatrixXd A = Eigen::MatrixXd::Identity(6, 6); // 状态转移矩阵A
            A.topRightCorner(3, 3) = dt * I;

            Eigen::MatrixXd B(6, 3); // 控制输入矩阵B
            B << 0.5 * dt * dt * I, dt * I; //作符会自动处理矩阵大小的匹配，因此它将按行或按列将值逐一插入矩阵 B。

            Eigen::MatrixXd C = Eigen::MatrixXd::Zero(3, 6); // 观测矩阵C
            C.leftCols(3) = I;

            // Compute process and measurement noise covariance
            // 计算过程和测量噪声协方差
            Eigen::MatrixXd Q = proc_var_ * B * B.transpose(); // 过程噪声协方差Q
            //! proc_var_是相当于过程噪声的方差，在有控制输入的时候，Q = wBBT，没有的时候，Q = wI， w就表示噪声的方差，其均值是0，高斯噪声
            Eigen::MatrixXd R = meas_var_ * I; // 测量噪声协方差R，测量噪声就和状态方程无关了，就单独成单位阵就可以了

            Eigen::Vector3d u = Eigen::Vector3d::Zero(); // 输入控制u
            if (active_) {
                u = gravity_; // 如果处于激活状态，则考虑重力加速度
            }

            // Predict new state
            // 预测新状态
            // We know that the ball cannot penetrate the floor, so we don't
            // let it
            // 我们知道球不能穿透地板，所以我们不允许它这样做
            kf::GaussianEstimate prediction =
                kf::predict(estimate_, A, Q, B * u); // 预测新的状态
            if (prediction.x(2) <= 0) {
                prediction.x(2) = 0; // 防止预测位置低于地面
            }
            //! 上一时刻的估计，就是上一时刻信任的状态量

            // Update the estimate if measurement falls within likely range
            // 如果测量值落在合理的范围内，则更新估计
            //! 当前时刻的测量值，还没有被信任，所以是y
            const double nis = kf::nis(prediction, C, R, y); // 计算NIS
            if (nis >= nis_bound_) {
                ROS_WARN_STREAM("Rejected position = " << y.transpose()); // 超出NIS范围则拒绝该测量
                estimate_ = prediction; // 使用预测结果作为新估计
                return;
            } else {
                estimate_ = kf::correct(prediction, C, R, y); // 更新估计
                //! 更新之后，利用预测的和测量的来计算新的估计，此时刻信任的状态量
            }
        }

        t_prev_ = t; // 更新上一次的时间戳；为了下一次计算时间步长
        ++msg_count_; // 增加消息计数
    }

   private:
    // Store last received time and configuration for numerical differentiation
    // 存储上次接收到的时间戳和配置，用于数值微分
    double t_prev_; // 上一次的时间戳

    // Process and noise variance
    // 过程和噪声方差
    double proc_var_; // 过程噪声方差
    double meas_var_; // 测量噪声方差

    Eigen::Vector3d gravity_; // 重力加速度

    // State estimate and Kalman filter
    // 状态估计和卡尔曼滤波器
    kf::GaussianEstimate estimate_; // 当前的状态估计

    bool active_ = false; // 是否处于激活状态

    // Height above which the object is "activated" and considered to be
    // undergoing projectile motion
    // 物体高于此高度时被“激活”并被认为正在进行抛物运动
    double activation_height_; // 激活高度

    // Height below which the object is "deactivated" and considered to be done
    // with or about to be done with projectile motion (i.e. it is about to fit
    // the ground)
    // 物体低于此高度时被“去激活”并被认为已完成或即将完成抛物运动（即它即将接触地面）
    double deactivation_height_; // 去激活高度

    // Measurement is rejected if NIS is above this bound
    // 如果NIS超出此界限，则拒绝测量结果
    double nis_bound_; // NIS边界值

    // Number of messages received
    // 接收到的消息数量
    size_t msg_count_ = 0; // 接收到的消息数量

};  // class ProjectileViconEstimator

class ProjectileViconEstimatorNode { // 定义ProjectileViconEstimatorNode类
   public:
    ProjectileViconEstimatorNode() {}

    void init(ros::NodeHandle& nh) {
        Eigen::Vector3d gravity(0, 0, -9.81); // 重力加速度

        // Load parameters
        // 加载参数
        std::string vicon_object_name;
        double proc_var, meas_var, nis_bound, activation_height, deactivation_height;
        nh.param<std::string>("/projectile/vicon_object_name",
                              vicon_object_name, "ThingProjectile"); // 加载Vicon对象名称参数
        nh.param<double>("/projectile/proc_var", proc_var, 1.0); // 加载过程噪声方差参数
        nh.param<double>("/projectile/meas_var", meas_var, 1e-4); // 加载测量噪声方差参数
        nh.param<double>("/projectile/activation_height", activation_height,
                         1.0); // 加载激活高度参数
        nh.param<double>("/projectile/deactivation_height", deactivation_height,
                         0.2); // 加载去激活高度参数

        ROS_INFO_STREAM("projectile proc var = " << proc_var); // 输出过程噪声方差信息
        ROS_INFO_STREAM("projectile meas var = " << meas_var); // 输出测量噪声方差信息

        // 14.156 corresponds to 3-sigma bound for 3-dim Gaussian variable
        // 14.156 对应于三维高斯变量的3倍标准差边界
        nh.param<double>("/projectile/nis_bound", nis_bound, 14.156); // 加载NIS边界值参数
        /*
        对于一个n-维的高斯分布（即多维的情况），NIS 的值服从卡方分布（Chi-squared distribution）。
        特别地，在卡尔曼滤波中，NIS 是一个卡方分布的随机变量。卡方分布的自由度通常与测量向量的维度n 相关。
        对于n=3（三维高斯变量），NIS 的值应该符合 卡方分布，其自由度为 3。卡方分布的 3-σ 边界（即95%的置信区间对应的值）大约是 7.815。
        然而，题目中提到的是 14.156，这实际上对应的是 99% 置信区间的边界。在卡方分布中，当自由度为 3 时，99% 的置信区间对应的临界值大约为 14.156。
        */

        const std::string vicon_topic =
            "/vicon/" + vicon_object_name + "/" + vicon_object_name; // 构造Vicon话题名
        vicon_sub_ = nh.subscribe(
            vicon_topic, 1, &ProjectileViconEstimatorNode::vicon_cb, this); // 订阅Vicon数据

        // This is a topic rather than a service, because we want to use this
        // from simulation while use_sim_time=True
        // 这是一个话题而不是服务，因为我们希望在use_sim_time=True的情况下从仿真中使用它
        reset_sub_ =
            nh.subscribe("/projectile/reset_estimate", 1,
                         &ProjectileViconEstimatorNode::reset_cb, this); // 订阅重置估计的话题

        joint_states_pub_ = nh.advertise<sensor_msgs::JointState>(
            "/projectile/joint_states", 1); // 发布关节状态

        estimator_.init(nh, proc_var, meas_var, nis_bound, activation_height,
                        deactivation_height, gravity); // 初始化估算器； 仅仅初始化，对一些值进行赋值
    }

    // Spin, publishing the most recent estimate at the specified rate.
    // 旋转，以指定速率发布最新的估计
    void spin(ros::Rate& rate) {
        // Wait until the estimator has enough information
        // 等待直到估算器有足够的信息
        while (ros::ok() && !estimator_.ready()) {
            ros::spinOnce();
            rate.sleep();
        } //等待获取足够的测量值

        while (ros::ok()) {
            ros::spinOnce(); //! 只有这个每次才能调用回调函数，即启动vicon_cb函数，更新状态估计，而后用下面的函数发布出去
            publish_joint_states(estimator_.q(), estimator_.v()); // 发布关节状态
            rate.sleep();
        }
    }

   private:
    void reset_cb(const std_msgs::Empty&) {
        estimator_.reset(); // 重置估算器
        ROS_INFO("Projectile estimate reset."); // 输出重置信息
    }

    void vicon_cb(const geometry_msgs::TransformStamped& msg) {
        // Get the current joint configuration
        // 获取当前关节配置
        double t = msg.header.stamp.toSec(); // 获取当前时间戳
        Eigen::Vector3d y;
        y << msg.transform.translation.x, msg.transform.translation.y,
            msg.transform.translation.z; // 提取位置信息

        //! 这里就包含了卡尔曼滤波器的更新步骤，从状态方程的定义，到预测xk，利用xk的测量值，更新xk的可信任估计值
        estimator_.update(t, y); // 更新估算器;

        //! 每收到一次vicon的数据，就更新一次状态估计
    }

    void publish_joint_states(const Eigen::Vector3d& q,
                              const Eigen::Vector3d& v) {
        sensor_msgs::JointState msg;
        msg.header.stamp = ros::Time::now(); // 设置消息时间戳
        msg.name = {"x", "y", "z"}; // 关节名称

        for (int i = 0; i < 3; ++i) {
            msg.position.push_back(q(i)); // 设置位置信息
            msg.velocity.push_back(v(i)); // 设置速度信息
        }

        joint_states_pub_.publish(msg); // 发布关节状态消息
    }

    // Subscribers to Vicon pose and reset hook. hook一般指回调函数
    // 订阅Vicon姿态和重置钩子
    ros::Subscriber vicon_sub_; // Vicon数据订阅者
    ros::Subscriber reset_sub_; // 重置命令订阅者

    // Publisher for position and velocity.
    // 发布位置和速度
    ros::Publisher joint_states_pub_; // 关节状态发布者

    // Underlying estimator
    // 内部使用的估算器实例
    ProjectileViconEstimator estimator_;
};

class ProjectileROSInterface { // 定义ProjectileROSInterface类
 //! 就只是负责订阅vicon发布的tf信息，存放在变量q和v中，并且提供ready()函数，判断是否已经接收到关节状态
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 确保Eigen对象正确对齐

    ProjectileROSInterface() {}

    void init(ros::NodeHandle& nh) {
        joint_state_sub_ =
            nh.subscribe("/projectile/joint_states", 1,
                         &ProjectileROSInterface::joint_state_cb, this); // 订阅关节状态
    }

    bool ready() const { return joint_states_received_; } // 判断是否接收到关节状态

    Eigen::Vector3d q() const { return q_; } // 获取位置信息

    Eigen::Vector3d v() const { return v_; } // 获取速度信息

   private:
    void joint_state_cb(const sensor_msgs::JointState& msg) {
        for (int i = 0; i < 3; ++i) {
            q_[i] = msg.position[i]; // 提取位置信息
            v_[i] = msg.velocity[i]; // 提取速度信息
        }
        joint_states_received_ = true; // 标记已接收到关节状态
    }

    bool joint_states_received_ = false; // 是否接收到关节状态标志

    ros::Subscriber joint_state_sub_; // 关节状态订阅者

    Eigen::Vector3d q_; // 位置信息
    Eigen::Vector3d v_; // 速度信息
};

}  // namespace tongbot



