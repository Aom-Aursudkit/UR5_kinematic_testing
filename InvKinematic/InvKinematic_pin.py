import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np

# ----------------------------
# Load UR5 model
# ----------------------------
urdf_path = "/home/aom/Documents/GitHub/UR5_kinematic_testing/src/Universal_Robots_ROS2_Description/urdf/ur5e.urdf"
model_path = "/home/aom/Documents/GitHub/UR5_kinematic_testing/src/Universal_Robots_ROS2_Description"


robot = RobotWrapper.BuildFromURDF(urdf_path, model_path)

model = robot.model
data = robot.data

# End-effector frame name (ROS UR5 uses "tool0")
ee_frame = model.getFrameId("tool0")


# ----------------------------
# Forward Kinematics
# ----------------------------
def fk(q):
    """
    Returns SE3 transform of UR5 end-effector for joint angle vector q.
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[ee_frame]


# ----------------------------
# IK (Gauss–Newton)
# ----------------------------
def ik(robot, target_pos, target_rot, 
                 q0, max_iter, eps, damping):

    model = robot.model
    data = robot.data

    if q0 is None:
        q = pin.neutral(model)
    else:
        q = q0.copy()

    ee_id = model.getFrameId("tool0")  # UR robots always have tool0

    for i in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        # Current pose
        oMf = data.oMf[ee_id]
        pos = oMf.translation
        rot = oMf.rotation

        # Errors
        err_pos = target_pos - pos
        err_rot = 0.5 * pin.log3(target_rot @ rot.T)  # rotation error

        err = np.concatenate((err_pos, err_rot))

        if np.linalg.norm(err) < eps:
            return q, True

        # Jacobian
        J = pin.computeFrameJacobian(model, data, q, ee_id, pin.LOCAL)
        J = np.vstack((J[0:3, :], J[3:6, :]))  # spatial jacobian

        # Dampened least squares
        H = J.T @ J + damping * np.eye(model.nq)
        g = J.T @ err

        dq = np.linalg.solve(H, g)
        q += dq

        # Enforce joint limits
        q = np.minimum(q, model.upperPositionLimit)
        q = np.maximum(q, model.lowerPositionLimit)

    return q, False


# ----------------------------
# Test
# ----------------------------

# Desired target pose
target_p = np.array([0.4, 0.4, 0.2])
# target_R = np.array([[0,  1,  0],
#                      [1,  0,  0],
#                      [0,  0,  -1]])
target_R = np.array([[1,  0,  0],
                     [0,  1,  0],
                     [0,  0,  1]])

# Elbow-up guessS
q_guess = np.array([0.0, -np.pi/4, np.pi/2, -np.pi/4, -np.pi/2, 0.0])

q_sol, success = ik(robot, target_p, target_R, q0=q_guess, max_iter=1000, eps=1e-4, damping=1e-2)

print("IK success:", success)
print("joint angles (radians):\n", q_sol)

# Verify FK
Tsol = fk(q_sol)
print("\nFK reached translation:", Tsol.translation)
print("FK reached rotation:\n", Tsol.rotation)
