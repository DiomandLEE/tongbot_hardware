"""Unit tests for the ros_utils component of the module."""
import pytest
import numpy as np
from spatialmath.base import rotx, r2q

from tongbot_hardware import ros_utils


def _qrotx(angle):
    """Quaternion corresponding to a rotation of `angle` radians about the x-axis.

    Quaternion is formatted [x, y, z, w].
    """
    return np.roll(r2q(rotx(angle)), -1)


def test_lerp_list_constant():
    """Test linear interpolation of a list of constant values."""
    # original data
    times2 = [0, 1, 2, 3]
    values2 = np.ones(4)

    # interpolated data
    times1 = [0.5, 1.5, 2.5]
    values1 = ros_utils.interpolate_list(times1, times2, values2)

    assert np.allclose(values1, np.ones(3))


def test_lerp_list_linear():
    """Test linear interpolation of a list of linearly increasing values."""
    # original data
    times2 = [0, 1, 2, 3]
    values2 = [0, 1, 2, 3]

    # interpolated data
    times1 = [0.5, 1.5, 2.5]
    values1 = ros_utils.interpolate_list(times1, times2, values2)

    assert np.allclose(values1, [0.5, 1.5, 2.5])

    # again but with negative values
    times2 = [0, 1, 2, 3]
    values2 = [0, -1, -2, -3]

    # interpolated data
    times1 = [0.5, 1.5, 2.5]
    values1 = ros_utils.interpolate_list(times1, times2, values2)

    assert np.allclose(values1, [-0.5, -1.5, -2.5])


def test_lerp_list_beyond_bounds():
    """Test linear interpolation of a list with times beyond its own range."""
    # original data
    times2 = [0, 1, 2, 3]
    values2 = [0, 1, 2, 3]

    # interpolated data
    # values should not be interpolated beyond the end of the list (since we
    # don't know what would be next)
    times1 = [-0.5, 0.5, 1.5, 2.5, 3.5]
    values1 = ros_utils.interpolate_list(times1, times2, values2)

    assert np.allclose(values1, [0, 0.5, 1.5, 2.5, 3])


def test_lerp_list_nonlinear():
    """Test linear interpolation of a list values with different step sizes."""
    # original data
    # note the jump from value 2 to 10
    times2 = [0, 1, 2, 3]
    values2 = [0, 1, 2, 10]

    # interpolated data
    times1 = [0.5, 1.5, 2.5]
    values1 = ros_utils.interpolate_list(times1, times2, values2)
    assert np.allclose(values1, [0.5, 1.5, 6])

    # original data
    # now we have a decrease in the values
    times2 = [0, 1, 2, 3]
    values2 = [0, 1, 2, 0]

    # interpolated data
    times1 = [0.5, 1.5, 2.5]
    values1 = ros_utils.interpolate_list(times1, times2, values2)
    assert np.allclose(values1, [0.5, 1.5, 1])


def test_lerp_list_vector():
    """Test linear interpolation of a list values with different step sizes."""
    # original data
    # note the jump from value 2 to 10
    times2 = [0, 1, 2]
    values2 = np.array([[0, 0, 0], [1, 2, 3], [2, 4, 6]])

    # interpolated data
    times1 = [0.5, 1.5]
    values1 = ros_utils.interpolate_list(times1, times2, values2)
    assert np.allclose(values1, [[0.5, 1, 1.5], [1.5, 3, 4.5]])


def test_slerp():
    """Test spherical linear interpolation (of quaternions)."""
    q0 = np.array([0, 0, 0, 1])
    q1 = np.array([1, 0, 0, 1]) / np.sqrt(2)  # 90 deg rotation about x
    s = 0.5

    q = ros_utils.slerp(q0, q1, s)
    qd = _qrotx(np.pi / 4)

    assert np.allclose(q, qd)


def test_slerp_list():
    """Test spherical linear interpolation at a list of values."""
    q0 = np.array([0, 0, 0, 1])
    q1 = np.array([1, 0, 0, 1]) / np.sqrt(2)  # 90 deg rotation about x
    s = [0, 0.25, 0.5, 0.75, 1]

    qs = ros_utils.interpolate_list(s, [0, 1], [q0, q1], method="slerp")
    qds = [_qrotx(0.5 * x * np.pi) for x in s]
    assert np.allclose(qs, qds)
