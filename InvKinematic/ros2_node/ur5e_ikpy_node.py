import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

from ikpy.chain import Chain


class IKPyUR5eNode(Node):
    def __init__(self):
        super().__init__('ur5e_ikpy_node')

        # ----------------------------------------------
        # Parameters
        # ----------------------------------------------
        self.declare_parameter('urdf_path', '')
        urdf_path = self.get_parameter('urdf_path').get_parameter_value().string_value

        if urdf_path == "":
            self.get_logger().error("No URDF path provided! Set param: urdf_path")
            rclpy.shutdown()
            return

        self.get_logger().info(f"Loading URDF: {urdf_path}")

        # ----------------------------------------------
        # Load chain
        # ----------------------------------------------
        self.robot_chain = Chain.from_urdf_file(urdf_path)

        # Mark only revolute joints active
        for i, link in enumerate(self.robot_chain.links):
            self.robot_chain.active_links_mask[i] = (link.bounds is not None)

        # ----------------------------------------------
        # Publishers & Subscribers
        # ----------------------------------------------
        self.joint_pub = self.create_publisher(JointState, 'ik_joint_states', 10)
        self.pose_sub = self.create_subscription(
            PoseStamped,
            'target_pose',
            self.callback_target_pose,
            10
        )

        self.get_logger().info("IKPy UR5e node is ready.")

        # Initial guess
        self.initial_guess = [
            0.0,
            0.0,
            0.0,
            -np.pi/4,
            np.pi/2,
            -np.pi/2,
            np.pi/2,
            0.0,
            0.0,
        ]

        # TCP offset
        self.tool_offset = np.array([0.0, 0.0, 0.05])


    # ============================================================
    # IK callback
    # ============================================================
    def callback_target_pose(self, msg: PoseStamped):
        # Extract target pose
        p = msg.pose.position
        o = msg.pose.orientation

        # Convert quaternion to rotation matrix
        rot = self.quaternion_to_matrix(o.x, o.y, o.z, o.w)

        # Apply tool offset
        target_pos = np.array([p.x, p.y, p.z]) + self.tool_offset

        # Build target frame
        target_frame = np.eye(4)
        target_frame[:3, :3] = rot
        target_frame[:3, 3] = target_pos

        # Solve IK
        joint_angles_full = self.robot_chain.inverse_kinematics_frame(
            target=target_frame,
            initial_position=self.initial_guess,
            orientation_mode="all"
        )

        # UR5e active joints are indices 2–7
        active_joints = joint_angles_full[2:8]

        # Publish JointState
        msg_js = JointState()
        msg_js.header.stamp = self.get_clock().now().to_msg()
        msg_js.name = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        msg_js.position = active_joints.tolist()
        self.joint_pub.publish(msg_js)

        # Log
        self.get_logger().info(f"Published IK solution: {active_joints}")


    # ============================================================
    # Helper: quaternion → rotation matrix
    # ============================================================
    def quaternion_to_matrix(self, x, y, z, w):
        R = np.zeros((3, 3))
        R[0, 0] = 1 - 2*(y*y + z*z)
        R[0, 1] = 2*(x*y - z*w)
        R[0, 2] = 2*(x*z + y*w)

        R[1, 0] = 2*(x*y + z*w)
        R[1, 1] = 1 - 2*(x*x + z*z)
        R[1, 2] = 2*(y*z - x*w)

        R[2, 0] = 2*(x*z - y*w)
        R[2, 1] = 2*(y*z + x*w)
        R[2, 2] = 1 - 2*(x*x + y*y)
        return R


def main(args=None):
    rclpy.init(args=args)
    node = IKPyUR5eNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
