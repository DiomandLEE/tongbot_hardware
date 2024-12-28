#pragma once

#include <math.h>

namespace tongbot {

inline double wrap_to_pi(double angle) {
    while (angle >= M_PI) {
        angle -= 2 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2 * M_PI;
    }
    return angle;
}

}  // namespace tongbot
