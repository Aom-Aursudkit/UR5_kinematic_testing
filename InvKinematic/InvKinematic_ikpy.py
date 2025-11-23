from ikpy.chain import Chain
import numpy as np

# ----------------------------
# Load UR5e
# ----------------------------
urdf_path = "/home/aom/Documents/GitHub/UR5_kinematic_testing/src/Universal_Robots_ROS2_Description/urdf/ur5e.urdf"
robot_chain = Chain.from_urdf_file(urdf_path)

# ----------------------------
# Automatically mark only revolute joints as active
# ----------------------------
for i, link in enumerate(robot_chain.links):
    if link.bounds is not None:  # revolute joints have bounds
        robot_chain.active_links_mask[i] = True
    else:
        robot_chain.active_links_mask[i] = False

# ----------------------------
# Target pose
# ----------------------------
target_position_temp = np.array([0.4, 0.4, 0.05]) # <---- position

target_R = np.array([[1, 0, 0], 
                     [0, 1, 0], 
                     [0, 0, 1]]) # <---- rotation

tool_offset = np.array([0.0, 0.0, 0.05])
target_position = target_position_temp + tool_offset
target_frame = np.eye(4)
target_frame[:3, :3] = target_R
target_frame[:3, 3] = target_position

# ----------------------------
# Initial guess
# ----------------------------
initial_position = [
    0.0,
    0.0,
    0.0,  # Joint 1 (base)
    -np.pi / 4,  # Joint 2 (shoulder lift)    → up
    np.pi / 2,  # Joint 3 (elbow)           → elbow up
    -np.pi / 2,  # Joint 4
    np.pi / 2,  # Joint 5
    0.0,  # Joint 6
    0.0,
]

# ----------------------------
# Solve IK
# ----------------------------
joint_angles = robot_chain.inverse_kinematics_frame(
    target=target_frame, initial_position=initial_position, orientation_mode="all"
)

# ----------------------------
# Check results
# ----------------------------
# print(robot_chain.active_links_mask)  # shows which links are active
# print(len(robot_chain.active_links_mask))

active_joint_angles = joint_angles[2:8]
print("joint angles (radians):\n", active_joint_angles)

fk_result = robot_chain.forward_kinematics(joint_angles)
print("\nFK reached translation:", fk_result[:3, 3])
print("FK reached rotation:\n", fk_result[:3, :3])


# ----------------------------
# Debug zone
# ----------------------------
# chain = Chain.from_urdf_file(urdf_path)

# print("\n=== LINK LIST ===")
# for i, link in enumerate(chain.links):
#     print(i, link.name, "(fixed =", getattr(link, "is_fixed", None), ")")

# print("Active links mask:", chain.active_links_mask)

