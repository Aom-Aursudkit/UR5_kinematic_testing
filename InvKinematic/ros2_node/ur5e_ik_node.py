import rclpy
from rclpy.node import Node

import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

# ================================
# Subproblem IK Helper Functions
# ================================
def rot(k, theta):
    k = k / np.linalg.norm(k)
    k_hat = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    return np.eye(3) + np.sin(theta)*k_hat + (1-np.cos(theta))*(k_hat @ k_hat)

def wrapToPi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def sp_1(p1, p2, k):
    KxP = np.cross(k, p1)
    A = np.column_stack([KxP, -np.cross(k, KxP)])
    x = A.T @ p2
    theta = np.arctan2(x[0], x[1])
    is_LS = abs(np.linalg.norm(p1)-np.linalg.norm(p2)) > 1e-8 or abs(np.dot(k,p1)-np.dot(k,p2)) > 1e-8
    return theta, is_LS

def sp_4(h, p, k, d):
    A_11 = np.cross(k, p)
    A_1 = np.column_stack([A_11, -np.cross(k, A_11)])
    A = h.T @ A_1
    b = d - h.T @ k * (k.T @ p)
    norm_A_2 = A @ A
    x_ls_tilde = A_1.T @ (h*b)
    if norm_A_2 > b**2:
        xi = np.sqrt(norm_A_2 - b**2)
        x_N_prime_tilde = np.array([A[1], -A[0]])
        sc1 = x_ls_tilde + x_N_prime_tilde * xi
        sc2 = x_ls_tilde - x_N_prime_tilde * xi
        theta = np.array([np.arctan2(sc1[0], sc1[1]), np.arctan2(sc2[0], sc2[1])])
        is_LS = False
    else:
        theta = np.arctan2(x_ls_tilde[0], x_ls_tilde[1])
        is_LS = True
    return theta, is_LS

def sp_3(p1, p2, k, d):
    return sp_4(p2, p1, k, 0.5*(np.dot(p1,p1)+np.dot(p2,p2)-d**2))

# ================================
# Forward Kinematics
# ================================
def fwdkin(kin, theta):
    p = kin.P[:,0].copy()
    R = np.eye(3)
    for i in range(len(kin.joint_type)):
        if kin.joint_type[i] == 0:
            R = R @ rot(kin.H[:,i], theta[i])
        p = p + R @ kin.P[:,i+1]
    return R, p

# ================================
# Analytic IK for UR5/UR5e
# ================================
def IK_UR(R_06, p_0T, kin):
    P = kin.P
    H = kin.H
    Q = []
    is_LS_vec = []
    p_06 = p_0T - P[:,0] - R_06 @ P[:,6]

    # θ1
    theta1, theta1_ls = sp_4(H[:,1], p_06, -H[:,0], H[:,1].T @ np.sum(P[:,1:5], axis=1))

    for q1 in np.atleast_1d(theta1):
        R_01 = rot(H[:,0], q1)

        # θ5
        theta5, theta5_ls = sp_4(H[:,1], H[:,5], H[:,4], H[:,1].T @ R_01.T @ R_06 @ H[:,5])

        for q5 in np.atleast_1d(theta5):
            R_45 = rot(H[:,4], q5)

            theta14, theta14_ls = sp_1(R_45 @ H[:,5], R_01.T @ R_06 @ H[:,5], H[:,1])
            q6, q6_ls = sp_1(R_45.T @ H[:,1], R_06.T @ R_01 @ H[:,1], -H[:,5])

            d_inner = R_01.T @ p_06 - P[:,1] - rot(H[:,1], theta14) @ P[:,4]
            d = np.linalg.norm(d_inner)

            theta3, theta3_ls = sp_3(-P[:,3], P[:,2], H[:,1], d)

            for q3 in np.atleast_1d(theta3):
                q2, q2_ls = sp_1(P[:,2] + rot(H[:,1], q3) @ P[:,3], d_inner, H[:,1])
                q4 = wrapToPi(theta14 - q2 - q3)

                q = np.array([q1, q2, q3, q4, q5, q6])
                Q.append(q)
                is_LS_vec.append(theta1_ls or theta5_ls or theta14_ls or theta3_ls or q2_ls or q6_ls)

    if len(Q) == 0:
        return np.zeros((6,0)), []

    return np.column_stack(Q), is_LS_vec

# ============================================================
# ROS2 Node
# ============================================================
class UR5eAnalyticIKNode(Node):
    def __init__(self):
        super().__init__("ur5e_analytic_ik_node")

        # Tool offset
        self.tool_offset = np.array([0.0, 0.0, 0.05])

        # Build Kin struct
        zv = np.array([0,0,0])
        ex = np.array([1,0,0])
        ey = np.array([0,1,0])
        ez = np.array([0,0,1])

        self.kin = type("Kin", (), {})()
        self.kin.H = np.column_stack([ez, -ey, -ey, -ey, -ez, -ey])
        self.kin.P = np.column_stack([
            0.1625*ez, zv, -0.425*ex, -0.3922*ex,
            -0.1333*ey - 0.0997*ez, zv, -0.0996*ey
        ])
        self.kin.joint_type = np.zeros(6)

        # ROS2 I/O
        self.pose_sub = self.create_subscription(
            PoseStamped, "/target_pose", self.pose_callback, 10
        )
        self.joint_pub = self.create_publisher(
            JointState, "/ik_joint_states", 10
        )

        self.get_logger().info("UR5e Analytic IK Node Ready!")

    # Quaternion → rotation matrix
    def quat_to_R(self, x, y, z, w):
        R = np.zeros((3,3))
        R[0,0] = 1 - 2*(y*y + z*z)
        R[0,1] = 2*(x*y - z*w)
        R[0,2] = 2*(x*z + y*w)

        R[1,0] = 2*(x*y + z*w)
        R[1,1] = 1 - 2*(x*x + z*z)
        R[1,2] = 2*(y*z - x*w)

        R[2,0] = 2*(x*z - y*w)
        R[2,1] = 2*(y*z + x*w)
        R[2,2] = 1 - 2*(x*x + y*y)
        return R

    # IK callback
    def pose_callback(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation

        R_target = self.quat_to_R(q.x, q.y, q.z, q.w)
        p_target = np.array([p.x, p.y, p.z]) + self.tool_offset

        Q, _ = IK_UR(R_target, p_target, self.kin)

        if Q.shape[1] == 0:
            self.get_logger().warn("No IK solution found!")
            return

        q_sol = Q[:,0]   # pick first solution

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        js.position = q_sol.tolist()
        self.joint_pub.publish(js)

        self.get_logger().info(f"Published IK solution: {q_sol}")


def main(args=None):
    rclpy.init(args=args)
    node = UR5eAnalyticIKNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
