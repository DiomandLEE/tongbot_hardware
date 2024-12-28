#pragma once

#include <ros/ros.h>
#include <Eigen/Eigen>

#include <geometry_msgs/Twist.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>

// include kinova kortex API
#include <kortex_driver/Base_ClearFaults.h>
#include <kortex_driver/CartesianSpeed.h>
#include <kortex_driver/BaseCyclic_Feedback.h>
#include <kortex_driver/ReadAction.h>
#include <kortex_driver/StopAction.h>
#include <kortex_driver/Stop.h>
#include <kortex_driver/ExecuteAction.h>
#include <kortex_driver/SetCartesianReferenceFrame.h>
#include <kortex_driver/CartesianReferenceFrame.h>
#include <kortex_driver/SendGripperCommand.h>
#include <kortex_driver/SendJointSpeedsCommand.h>
#include <kortex_driver/GripperMode.h>
#include <kortex_driver/ActionNotification.h>
#include <kortex_driver/ActionEvent.h>
#include <kortex_driver/ActionType.h>
#include <kortex_driver/GetMeasuredCartesianPose.h>
#include <kortex_driver/OnNotificationActionTopic.h>
#include <kortex_driver/ExecuteWaypointTrajectory.h>
#include <kortex_driver/ValidateWaypointList.h>
#include <kortex_driver/GetProductConfiguration.h>
#include <kortex_driver/ModelId.h>

namespace tongbot {

std::map<std::string, size_t> JOINT_INDEX_MAP = {
    {"joint_1", 0}, {"joint_2", 1},
    {"joint_3", 2},        {"joint_4", 3},
    {"joint_5", 4},      {"joint_6", 5},    {"joint_7", 6}};

// Done: make a corresponding Vicon object class， 在python文件中已经实现，但是没有完成C++的，好像没有用到
//todo： 需要添加一个指令：对gripper进行开闭操作

class RobotROSInterface {
   public:
    RobotROSInterface(size_t nq, size_t nv) : nq_(nq), nv_(nv) {
        q_ = Eigen::VectorXd::Zero(nq); //Position
        v_ = Eigen::VectorXd::Zero(nv); //Velocity 之后可以设置为0
    }

    size_t nq() const { return nq_; };
    size_t nv() const { return nv_; };

    virtual Eigen::VectorXd q() const { return q_; } //虚函数，子类override标志重写

    virtual Eigen::VectorXd v() const { return v_; }

    virtual void brake() { publish_cmd_vel(Eigen::VectorXd::Zero(nv_)); } //停止，设置速度为0

    virtual bool ready() const { return joint_states_received_; }

    virtual void publish_cmd_vel(const Eigen::VectorXd& cmd_vel,
                                 bool bodyframe = false) = 0; //纯虚函数，那么该类属于抽象类，无法实例化，子类也是在定义了纯虚函数之后，才可以实例化

   protected:
    size_t nq_;
    size_t nv_;

    Eigen::VectorXd q_;
    Eigen::VectorXd v_;

    bool joint_states_received_ = false;

    ros::Subscriber joint_state_sub_;
    ros::Publisher cmd_pub_;
};

//定义子类，改为Dingo
class DingoROSInterface : public RobotROSInterface {
   public:
    DingoROSInterface(ros::NodeHandle& nh) : RobotROSInterface(3, 3) {
        //如果父类有默认构造函数，并且子类的构造函数中没有显式调用父类的构造函数，编译器会自动调用父类的默认构造函数。
        //如果父类没有默认构造函数，子类的构造函数必须显式调用父类的一个构造函数，并传递适当的参数。
        joint_state_sub_ =
            nh.subscribe("dingo/joint_states", 1,
                         &DingoROSInterface::joint_state_cb, this);//类内可以使用私有函数

        cmd_pub_ =
            nh.advertise<geometry_msgs::Twist>("dingo/cmd_vel", 1, true);
    }

    //子类定义纯虚函数
    //给一个控制指令，转化为相应的frame下的控制指令
    void publish_cmd_vel(const Eigen::VectorXd& cmd_vel,
                         bool bodyframe = false) override {
        // bodyframe indicates whether the supplied command is in the body
        // frame or world frame. The robot takes commands in the body frame, so
        // if the command is in the world it must be rotated.
        // bodyframe 表示提供的命令是以机体坐标系还是世界坐标系给出的。
        // 机器人接收的命令是以机体坐标系为基础的，因此如果命令是以世界坐标系给出的，则必须进行旋转
        if (cmd_vel.rows() != nv_) {
            throw std::runtime_error("Dingo given cmd_vel of wrong shape.");
        }

        geometry_msgs::Twist msg;
        if (bodyframe) {
            msg.linear.x = cmd_vel(0);
            msg.linear.y = cmd_vel(1);
        } else /* world frame */ {
            // we have to rotate into the body frame
            Eigen::Rotation2Dd C_bw(-q_(2)); //body系到world系。乘以world系下的数据，可以换到body系
            //用于表示二维旋转。它提供了对 2D 空间中的旋转进行处理的功能。Eigen::Rotation2Dd 是一个以弧度表示旋转角度的旋转矩阵。
            Eigen::Vector2d xy = C_bw * cmd_vel.head(2); //w-Trans-b 最终的frame在左面
            msg.linear.x = xy(0);
            msg.linear.y = xy(1);
        }
        msg.angular.z = cmd_vel(2);
        cmd_pub_.publish(msg);
    }

   private:
    void joint_state_cb(const sensor_msgs::JointState& msg) {
        for (int i = 0; i < msg.name.size(); ++i) {
            q_[i] = msg.position[i];
            v_[i] = msg.velocity[i];
        }
        joint_states_received_ = true; //就是第一次过来就变为true，收到了设备的state，就是ready了
    }//回调函数
};

class KinovaROSInterface : public RobotROSInterface {
   public:
    KinovaROSInterface(ros::NodeHandle& nh) : RobotROSInterface(7, 7) {
        std::string robot_name = "my_gen3";

        last_action_notification_event = 0;
        last_action_notification_id = 0;
        // Service --> Client
        ActivateNotif_ = nh.serviceClient<kortex_driver::OnNotificationActionTopic>("/" + robot_name + "/base/activate_publishing_of_action_topic");
        ClearFaults_ = nh.serviceClient<kortex_driver::Base_ClearFaults>("/" + robot_name + "/base/clear_faults");
        SendJointSpeedsCommand_ = nh.serviceClient<kortex_driver::SendJointSpeedsCommand>("/" + robot_name + "/base/send_joint_speeds_command");
        StopAction_ = nh.serviceClient<kortex_driver::Stop>("/" + robot_name + "/base/stop");

        joint_state_sub_ =
            nh.subscribe("/" + robot_name + "/joint_states", 1,
                            &KinovaROSInterface::joint_state_cb, this);//类内可以使用私有函数
        cmd_pub_ =
            nh.advertise<std_msgs::Float64MultiArray>("/my_gen3/cmd_vel", 1, true);
        //这是为了在录包的时候，把控制指令录进去
        notif_sub_ =
            nh.subscribe("/" + robot_name + "/action_topic", 1000,
                            &KinovaROSInterface::notification_callback, this);

        // Activate Notification for KINOVA
        action_notif();
        // Clear Faults for KINOVA
        clear_faults();
    }


    bool wait_for_action_end_or_abort(){
        while (ros::ok()){
            // load应该就是查看action_notification的值，一开始是给他设置为0的
            if (last_action_notification_event.load() == kortex_driver::ActionEvent::ACTION_END){
                ROS_INFO("Kinova Received ACTION_END notification for action %d", last_action_notification_id.load());
                return true;
            }else if (last_action_notification_event.load() == kortex_driver::ActionEvent::ACTION_ABORT){
                ROS_ERROR("Kinova Received ACTION_ABORT notification for action %d", last_action_notification_id.load());
                // all_notifs_succeeded = false;
                return false;
                }
        ros::spinOnce();
        }
        // 如果不可以的话，尝试在这里使用spin，然后把上面的都注释掉
        // ros::spinOnce();
        return false;
    }

    void publish_cmd_vel(const Eigen::VectorXd& cmd_vel,
                         bool bodyframe = false) override {
        if (cmd_vel.rows() != nv_) {
            throw std::runtime_error("Kinova given cmd_vel of wrong shape.");
        }

        std_msgs::Float64MultiArray msg;
        msg.data = std::vector<double>(cmd_vel.data(),
                                       cmd_vel.data() + cmd_vel.rows());
        cmd_pub_.publish(msg); //发布指令，为了rosbag

        msgBaseJointSpeeds_.joint_speeds.clear(); //清空前一个JointSpeeds
        for (size_t i = 0; i < cmd_vel.size(); ++i){
            msgJointSpeed_.joint_identifier = i; // joint_i
            msgJointSpeed_.duration = 0;
            msgJointSpeed_.value = cmd_vel[i];
            msgBaseJointSpeeds_.joint_speeds.push_back(msgJointSpeed_);
        }

        srvSendJointSpeeds_.request.input.joint_speeds = msgBaseJointSpeeds_.joint_speeds;
        //call server
        last_action_notification_event = 1;
        try{
            SendJointSpeedsCommand_.call(srvSendJointSpeeds_);
            ROS_INFO("Kinova JOINTS SPEEDS has send!!!");
        }catch(...){
            std::string error_string = "Kinova Failed to call Command on send joints speeds ...";
            ROS_ERROR("%s", error_string.c_str());
            throw;
        }

        wait_for_action_end_or_abort(); //这个之后不行的话可以去掉。
    }

   private:
    void joint_state_cb(const sensor_msgs::JointState& msg) {
        for (int i = 0; i < msg.name.size(); ++i) {
            size_t j = JOINT_INDEX_MAP.at(msg.name[i]); //这个挺有意思 //找joint name对应 index
            q_[j] = msg.position[i];
            v_[j] = msg.velocity[i];
        }
        joint_states_received_ = true;
    }

    // kinova中对执行的动作的通知
    bool action_notif(){
        // 获取action_notification
        if (ActivateNotif_.call(srvActivateNotif_)){
            ROS_INFO("Kinova Action notification activated!");
            return true;
        }else{
            std::string error_string = "Kinova Action notification publication failed";
            ROS_ERROR("%s", error_string.c_str());
            return false;}
    }

    // 清除之前残留的故障fault
    bool clear_faults(){
        // Clear the faults
        if (!ClearFaults_.call(srvClearFaults_)){
            std::string error_string = "Kinova Failed to clear the faults";
            ROS_ERROR("%s", error_string.c_str());
            return false;}

        ros::Duration(0.1).sleep();
        ROS_INFO("KINOVA Previous Faults Have cleared");
        return true;
    }

    // 将弧度转化为角度
    double rad2deg(double rad){
        return rad * 180 / M_PI;
    }

    void notification_callback(const kortex_driver::ActionNotification& notif){
        last_action_notification_event = notif.action_event;
        last_action_notification_id = notif.handle.identifier;
    }

    //todo notification_callback(), JoiintNameMap{}, SendJointSpeeds

    //创建客户端
    ros::ServiceClient ClearFaults_, SendJointSpeedsCommand_, StopAction_, ActivateNotif_;

    // 发布Joint Speeds
    kortex_driver::OnNotificationActionTopic srvActivateNotif_;
    kortex_driver::Base_ClearFaults srvClearFaults_;
    kortex_driver::SendJointSpeedsCommand srvSendJointSpeeds_;
    kortex_driver::Stop srvStopAction_;

    // srv中的msg
    // kortex_driver::ConstrainedJointAngles _constrained_joint_angles;
    // kortex_driver::JointAngles _joint_angles;
    // kortex_driver::JointAngle _joint_angle;
    kortex_driver::Base_JointSpeeds msgBaseJointSpeeds_;
    kortex_driver::JointSpeed msgJointSpeed_;

    std::atomic<int> last_action_notification_event;
    std::atomic<int> last_action_notification_id;
    ros::Subscriber notif_sub_;
};

class TongbotROSInterface : public RobotROSInterface {
   public:
    TongbotROSInterface(ros::NodeHandle& nh)
        : base_(nh), arm_(nh), RobotROSInterface(10, 10) {}

    Eigen::VectorXd q() const override {
        Eigen::VectorXd q(nq_);
        q << base_.q(), arm_.q();
        return q;
    } //重写

    Eigen::VectorXd v() const override {
        Eigen::VectorXd v(nv_);
        v << base_.v(), arm_.v();
        return v;
    } //重写

    void brake() override {
        base_.brake();
        arm_.brake();
    }

    bool ready() const override { return base_.ready() && arm_.ready(); }

    void publish_cmd_vel(const Eigen::VectorXd& cmd_vel,
                         bool bodyframe = false) override {
        base_.publish_cmd_vel(cmd_vel.head(base_.nv()), bodyframe);
        arm_.publish_cmd_vel(cmd_vel.tail(arm_.nv()));
    }

   private:
    DingoROSInterface base_;
    KinovaROSInterface arm_;
};

}  // namespace tongbot
