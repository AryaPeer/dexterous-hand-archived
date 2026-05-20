import numpy as np
from numpy.testing import assert_allclose

from dexterous_hand.utils.cpu.quaternion import (
    quat_angular_distance,
    quat_conjugate,
    quat_from_axis_angle,
    quat_multiply,
    quat_rotate_vector,
    quat_to_axis_angle,
    quat_to_rotation_matrix,
    random_quaternion,
    random_quaternion_within_angle,
)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
                             
Z90 = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
                              
Z180 = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi)
                             
X90 = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi / 2)

class TestQuatMultiply:
    def test_identity_left(self):
        result = quat_multiply(IDENTITY, Z90)
        assert_allclose(result, Z90, rtol=1e-6)

    def test_identity_right(self):
        result = quat_multiply(Z90, IDENTITY)
        assert_allclose(result, Z90, rtol=1e-6)

    def test_double_rotation(self):
                                                                    
        result = quat_multiply(Z90, Z90)
        assert_allclose(result, Z180, atol=1e-6)

    def test_non_commutative(self):
        q1q2 = quat_multiply(Z90, X90)
        q2q1 = quat_multiply(X90, Z90)
                                             
        assert not np.allclose(q1q2, q2q1, rtol=1e-6)

class TestQuatConjugate:
    def test_identity(self):
        result = quat_conjugate(IDENTITY)
        assert_allclose(result, IDENTITY, rtol=1e-6)

    def test_inverse_property(self):
                                                                   
        result = quat_multiply(Z90, quat_conjugate(Z90))
        assert_allclose(result, IDENTITY, rtol=1e-6)

    def test_signs(self):
        q = np.array([0.5, 0.3, 0.2, 0.1])
        q = q / np.linalg.norm(q)
        conj = quat_conjugate(q)
        assert_allclose(conj[0], q[0], rtol=1e-6)
        assert_allclose(conj[1:], -q[1:], rtol=1e-6)

class TestQuatAngularDistance:
    def test_identity_zero(self):
        dist = quat_angular_distance(IDENTITY, IDENTITY)
        assert_allclose(dist, 0.0, atol=1e-6)

    def test_90_degrees(self):
        dist = quat_angular_distance(IDENTITY, Z90)
        assert_allclose(dist, np.pi / 2, atol=1e-6)

    def test_180_degrees(self):
        dist = quat_angular_distance(IDENTITY, Z180)
        assert_allclose(dist, np.pi, atol=1e-4)

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        q1 = random_quaternion(rng)
        q2 = random_quaternion(rng)
        assert_allclose(
            quat_angular_distance(q1, q2),
            quat_angular_distance(q2, q1),
            rtol=1e-6,
        )

    def test_range(self):
        rng = np.random.default_rng(7)
        for _ in range(100):
            q1 = random_quaternion(rng)
            q2 = random_quaternion(rng)
            dist = quat_angular_distance(q1, q2)
            assert 0.0 <= dist <= np.pi + 1e-6

    def test_double_cover(self):
        rng = np.random.default_rng(12)
        q = random_quaternion(rng)
        r = random_quaternion(rng)
        d1 = quat_angular_distance(q, r)
        d2 = quat_angular_distance(-q, r)
        assert_allclose(d1, d2, rtol=1e-6)

    def test_triangle_inequality(self):
        rng = np.random.default_rng(55)
        for _ in range(50):
            a = random_quaternion(rng)
            b = random_quaternion(rng)
            c = random_quaternion(rng)
            dab = quat_angular_distance(a, b)
            dbc = quat_angular_distance(b, c)
            dac = quat_angular_distance(a, c)
            assert dac <= dab + dbc + 1e-6

class TestRandomQuaternion:
    def test_unit_norm(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            q = random_quaternion(rng)
            assert_allclose(np.linalg.norm(q), 1.0, rtol=1e-6)

    def test_distribution_coverage(self):
        rng = np.random.default_rng(1)
        ws = [random_quaternion(rng)[0] for _ in range(1000)]
                                                                              
        assert min(ws) < -0.8
        assert max(ws) > 0.8

class TestRandomQuaternionWithinAngle:
    def test_within_bound(self):
        rng = np.random.default_rng(42)
        for _ in range(500):
            q = random_quaternion_within_angle(rng, 0.5)
            _, angle = quat_to_axis_angle(q)
            assert angle <= 0.5 + 1e-6

    def test_unit_norm(self):
        rng = np.random.default_rng(77)
        for _ in range(100):
            q = random_quaternion_within_angle(rng, 1.0)
            assert_allclose(np.linalg.norm(q), 1.0, rtol=1e-6)

    def test_full_so3_fallback(self):
                                                                             
                                                                                
        rng = np.random.default_rng(33)
        angles = []
        for _ in range(200):
            q = random_quaternion_within_angle(rng, 2 * np.pi)
            assert_allclose(np.linalg.norm(q), 1.0, rtol=1e-6)
            _, angle = quat_to_axis_angle(q)
            angles.append(angle)
                                                  
        assert max(angles) > 2.5

class TestQuatToRotationMatrix:
    def test_identity(self):
        R = quat_to_rotation_matrix(IDENTITY)
        assert_allclose(R, np.eye(3), rtol=1e-6)

    def test_orthogonal(self):
        rng = np.random.default_rng(10)
        q = random_quaternion(rng)
        R = quat_to_rotation_matrix(q)
        assert_allclose(R @ R.T, np.eye(3), atol=1e-6)

    def test_determinant_one(self):
        rng = np.random.default_rng(20)
        q = random_quaternion(rng)
        R = quat_to_rotation_matrix(q)
        assert_allclose(np.linalg.det(R), 1.0, rtol=1e-6)

    def test_known_rotation(self):
                                                 
        R = quat_to_rotation_matrix(Z90)
        rotated = R @ np.array([1.0, 0.0, 0.0])
        assert_allclose(rotated, np.array([0.0, 1.0, 0.0]), atol=1e-6)

class TestQuatRotateVector:
    def test_identity_no_rotation(self):
        v = np.array([1.0, 2.0, 3.0])
        result = quat_rotate_vector(IDENTITY, v)
        assert_allclose(result, v, rtol=1e-6)

    def test_90_z(self):
                                                 
        result = quat_rotate_vector(Z90, np.array([1.0, 0.0, 0.0]))
        assert_allclose(result, np.array([0.0, 1.0, 0.0]), atol=1e-6)

    def test_180_z(self):
                                                   
        result = quat_rotate_vector(Z180, np.array([1.0, 0.0, 0.0]))
        assert_allclose(result, np.array([-1.0, 0.0, 0.0]), atol=1e-6)

class TestAxisAngleRoundtrip:
    def test_roundtrip(self):
        test_cases = [
            (np.array([1.0, 0.0, 0.0]), np.pi / 4),
            (np.array([0.0, 1.0, 0.0]), np.pi / 2),
            (np.array([0.0, 0.0, 1.0]), np.pi),
            (np.array([1.0, 1.0, 0.0]) / np.sqrt(2), 1.23),
        ]
        for axis_in, angle_in in test_cases:
            q = quat_from_axis_angle(axis_in, angle_in)
            axis_out, angle_out = quat_to_axis_angle(q)
            assert_allclose(angle_out, angle_in, rtol=1e-6)
            assert_allclose(axis_out, axis_in, atol=1e-6)

    def test_identity(self):
        q = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.0)
        assert_allclose(q, IDENTITY, atol=1e-6)

    def test_small_angle(self):
                                                       
        axis = np.array([0.0, 0.0, 1.0])
        angle = 1e-10
        q = quat_from_axis_angle(axis, angle)
        assert_allclose(np.linalg.norm(q), 1.0, rtol=1e-6)
        _, angle_out = quat_to_axis_angle(q)
        assert_allclose(angle_out, angle, atol=1e-6)
