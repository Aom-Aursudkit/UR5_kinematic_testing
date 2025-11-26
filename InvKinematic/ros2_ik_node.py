import numpy as np
from ur_analytic_ik import ur5e
import pinocchio as pin

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import pandas as pd


class UR5eAnalyticIK:

    def __init__(self, urdf_path):
        self.tool_extension = -0.059

        self.robot = pin.buildModelFromUrdf(urdf_path)
        self.data = self.robot.createData()

        # End-effector frame ID
        self.ee_name = "tool0"
        self.ee_id = self.robot.getFrameId(self.ee_name)

        np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

    def build_transform(self, p, R):
        T = np.eye(4)
        T[:3, 3] = p
        T[:3, :3] = R
        return T

    def apply_tool_extension(self, T):
        if self.tool_extension == 0.0:
            return T

        T_tool = np.eye(4)
        T_tool[2, 3] = self.tool_extension
        return T @ T_tool

    def solve_ik(self, p, R, elbow_up_only=True):
        T = self.build_transform(p, R)
        T_ext = self.apply_tool_extension(T)

        solutions = ur5e.inverse_kinematics(T_ext)

        if elbow_up_only:
            solutions = [q for q in solutions if q[1] < 0]

        return solutions, T_ext

    def jacobian(self, q):
        pin.computeJointJacobians(self.robot, self.data, q)
        pin.updateFramePlacements(self.robot, self.data)
        J = pin.getFrameJacobian(
            self.robot, self.data, self.ee_id, pin.ReferenceFrame.LOCAL
        )
        return J

    def solve_joint_velocities(self, q, ee_velocity):
        J = self.jacobian(q)
        Jp = J[:3, :]
        qdot = np.linalg.pinv(Jp) @ ee_velocity
        return qdot

    def solve_joint_accelerations(self, q, qdot, ee_acc, dt=1e-4):
        J = self.jacobian(q)

        q_next = q.copy()
        q_next += qdot * dt
        J_next = self.jacobian(q_next)

        Jdot = (J_next - J) / dt

        Jp = J[:3, :]
        Jpdot = Jdot[:3, :]

        qddot = np.linalg.pinv(Jp) @ (ee_acc - Jpdot @ qdot)
        return qddot

    def forward_kinematics(self, q):
        return ur5e.forward_kinematics(*q)


class JointSpacePublisher(Node):

    def __init__(self):
        super().__init__("joint_space_node")
        self.joint_publisher_ = self.create_publisher(
            Float64MultiArray, "joint_desire_pos", 10
        )
        self.vel_publisher_ = self.create_publisher(
            Float64MultiArray, "joint_desire_vel", 10
        )
        self.acc_publisher_ = self.create_publisher(
            Float64MultiArray, "joint_desire_acc", 10
        )

        urdf_path = "urdf path here"

        self.P = np.array([0.42, 0.42, 0.01])
        self.R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        df = pd.read_csv("csv path here")
        self.positions = df[["x", "y", "z"]].to_numpy()
        self.velocities = df[["vx", "vy", "vz"]].to_numpy()
        self.accelerations = df[["ax", "ay", "az"]].to_numpy()

        self.ik = UR5eAnalyticIK(urdf_path)
        self.solution_index = 0  # track which solution to use

        self.iteration = 0

        timer_period = 0.02
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        self.P = self.positions[-1]
        V = self.velocities[-1]
        A = self.accelerations[-1]

        self.R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        solutions, T_target = self.ik.solve_ik(self.P, self.R, elbow_up_only=True)
        solutions = solutions[0]

        qdot = self.ik.solve_joint_velocities(solutions, V)
        qddot = self.ik.solve_joint_accelerations(solutions, qdot, A)

        # if self.iteration < 1000:
        #     self.P = np.array([0.42, 0.42, 0.01])
        #     self.R = np.array([
        #         [0, 1, 0],
        #         [1, 0, 0],
        #         [0, 0,-1]
        #     ])
        #     self.get_logger().info(f"PEN-UP {self.iteration}")
        # elif self.iteration > 1000:
        #     self.P = np.array([0.42, 0.42, -0.03])
        #     self.R = np.array([
        #         [0, 1, 0],
        #         [1, 0, 0],
        #         [0, 0,-1]
        #     ])
        #     self.get_logger().info(f"PEN-DOWN {self.iteration}")

        # self.iteration += 1

        # if self.iteration > 2000:
        #     self.iteration = 0

        msg = Float64MultiArray()
        msg.data = solutions.tolist()
        self.joint_publisher_.publish(msg)

        vel_msg = Float64MultiArray()
        vel_msg.data = qdot.tolist()
        self.vel_publisher_.publish(vel_msg)

        acc_msg = Float64MultiArray()
        acc_msg.data = qddot.tolist()
        self.acc_publisher_.publish(acc_msg)


def main(args=None):

    rclpy.init(args=args)

    joint_space_publisher = JointSpacePublisher()

    rclpy.spin(joint_space_publisher)

    joint_space_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
