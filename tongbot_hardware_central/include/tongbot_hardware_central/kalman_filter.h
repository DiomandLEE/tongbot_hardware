#pragma once

#include <Eigen/Eigen>

namespace tongbot {
namespace kf {

// 定义一个结构体表示高斯估计，包括状态向量和协方差矩阵
struct GaussianEstimate {
    GaussianEstimate() {}  // 默认构造函数
    GaussianEstimate(const Eigen::VectorXd& x, const Eigen::MatrixXd& P) : x(x), P(P) {}  // 带参数的构造函数，初始化状态向量x和协方差矩阵P

    Eigen::VectorXd x;  // 状态向量，表示系统的状态（如位置、速度等）
    Eigen::MatrixXd P;  // 协方差矩阵，表示状态估计的不确定性

    //! 卡尔曼滤波的流程
    /*
    根据前一时刻的结果，计算预测的下一时刻状态和过程协方差矩阵，而后根据当前的测量值，对预测结果进行校正，得到新的状态估计和协方差矩阵。
    */
};

// 预测步骤：根据先前的状态和先验模型预测下一个状态
GaussianEstimate predict(const GaussianEstimate& e, const Eigen::MatrixXd& A,
                         const Eigen::MatrixXd& Q, const Eigen::VectorXd& v) {
    GaussianEstimate prediction;  // 创建一个新的GaussianEstimate对象来存储预测值

    // 预测状态向量：x' = A * x + v，其中A是状态转移矩阵，v是控制输入
    prediction.x = A * e.x + v; //这里的v就是B*u

    // 预测协方差矩阵：P' = A * P * A^T + Q，其中Q是过程噪声协方差矩阵
    // 状态转移过程中的不确定性的协方差P
    prediction.P = A * e.P * A.transpose() + Q;

    return prediction;  // 返回预测后的高斯估计
}

// 校正步骤：将测量值与预测的状态融合以更新估计
GaussianEstimate correct(const GaussianEstimate& e, const Eigen::MatrixXd& C,
                         const Eigen::MatrixXd& R, const Eigen::VectorXd& y) {
    // 其中y是测量值，C是测量矩阵，x是预测的状态
    // 残差协方差：计算矩阵S，用于计算卡尔曼增益
    Eigen::MatrixXd CP = C * e.P;  // 用测量矩阵C乘以预测协方差P
    Eigen::MatrixXd S = CP * C.transpose() + R;  // 加上测量噪声协方差R

    // 执行Cholesky分解来解决卡尔曼增益的计算
    Eigen::LLT<Eigen::MatrixXd> LLT = S.llt();  // 对矩阵S进行Cholesky分解，得到下三角矩阵

    // 使用测量模型来校正状态估计
    GaussianEstimate correction;  // 创建一个新的GaussianEstimate对象来存储校正后的状态

    // 更新协方差矩阵：P' = P - K * C * P
    correction.P = e.P - CP.transpose() * LLT.solve(CP);

    // 更新状态估计：x' = x + K * (y - C * x)
    correction.x = e.x + CP.transpose() * LLT.solve(y - C * e.x);

    return correction;  // 返回校正后的高斯估计
}

// 预测和校正步骤的组合：执行预测和校正
GaussianEstimate predict_and_correct(const GaussianEstimate& e,
                                     const Eigen::MatrixXd& A,
                                     const Eigen::MatrixXd& Q,
                                     const Eigen::VectorXd& v,
                                     const Eigen::MatrixXd& C,
                                     const Eigen::MatrixXd& R,
                                     const Eigen::VectorXd& y) {
    // 首先根据模型和控制输入进行状态预测
    GaussianEstimate prediction = predict(e, A, Q, v);

    // 然后使用测量值校正预测的状态
    return correct(prediction, C, R, y);  // 执行校正并返回结果
}

// 计算规范化残差平方（NIS），用于假设检验，判断是否拒绝不良测量
double nis(const GaussianEstimate& e, const Eigen::MatrixXd& C,
           const Eigen::MatrixXd& R, const Eigen::VectorXd& y) {
    // 残差：z = y - C * x，其中y是测量值，C是测量矩阵，x是预测的状态
    Eigen::VectorXd z = y - C * e.x; //! y是k时刻的测量，x是k-1时刻预测的k时刻的状态

    // 残差协方差：S = C * P * C^T + R，其中P是预测的协方差矩阵，R是测量噪声
    Eigen::MatrixXd S = C * e.P * C.transpose() + R;

    // 对残差协方差S进行Cholesky分解
    Eigen::LLT<Eigen::MatrixXd> LLT = S.llt();

    // 返回规范化残差平方值：z' * S^-1 * z
    return z.dot(LLT.solve(z));
    // 太大的话，就认为这个测量和估计是错误的，
}

}  // namespace kf
}  // namespace tongbot
