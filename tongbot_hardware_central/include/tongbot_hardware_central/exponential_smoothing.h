#pragma once

#include <ros/ros.h>

namespace tongbot {

// First-order exponential smoothing:
// https://en.wikipedia.org/wiki/Exponential_smoothing
template <typename T>
class ExponentialSmoother {
   public:
    ExponentialSmoother() {}//read 就是平滑，将新旧状态混合：state = c * measured + (1 - c) * prev

    // Initialize with time constant tau and initial value x0. As tau
    // increases, new measurements are trusted less and more weight is
    // given to the old states.
    //
    // Take care setting x0 since it can heavily weight early
    // estimates depending on the magnitude of tau.

    // 使用时间常数 tau 和初始值 x0 进行初始化。随着 tau 的增加，
    // 新的测量值被信任得更少，而旧的状态被赋予更多的权重。
    //
    // 设置 x0 时需要小心，因为它可能会根据 tau 的大小对早期估计产生很大的影响。

    void init(double tau, const T& x0) {
        this->tau = tau;
        prev = x0;
    }

    // Generate next estimate from measurement.
    T next(const T& measured, double dt) {
        double c = 1.0 - std::exp(-dt / tau);
        T state = c * measured + (1 - c) * prev;
        prev = state;
        return state;
    }

   private:
    T prev;      // previous measurement
    double tau;  // time constant
};               // class ExponentialSmoother

}  // namespace tongbot
