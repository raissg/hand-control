"""
Real-time hand tracking → InMoov robot hand animation.

Webcam → MediaPipe hand landmarks → finger bend angles → PyBullet URDF joints.

Usage:
    pip install mediapipe opencv-python pybullet numpy
    python hand_control.py

Press 'q' in the webcam window to quit.
"""

import math
import os
import time

import cv2
import mediapipe as mp
import numpy as np
import pybullet as p
import pybullet_data

# ---------------------------------------------------------------------------
# MediaPipe landmark indices
# ---------------------------------------------------------------------------
# Thumb:  1(CMC) 2(MCP) 3(IP) 4(TIP)
# Index:  5(MCP) 6(PIP) 7(DIP) 8(TIP)
# Middle: 9(MCP) 10(PIP) 11(DIP) 12(TIP)
# Ring:   13(MCP) 14(PIP) 15(DIP) 16(TIP)
# Pinky:  17(MCP) 18(PIP) 19(DIP) 20(TIP)

FINGER_LANDMARKS = {
    'thumb':  [1, 2, 3, 4],
    'index':  [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring':   [13, 14, 15, 16],
    'pinky':  [17, 18, 19, 20],
}

# 5 control joints in the URDF
CONTROL_JOINTS = {
    'thumb':  'r_thumb_joint',
    'index':  'r_index_joint',
    'middle': 'r_middle_joint',
    'ring':   'r_ring_joint',
    'pinky':  'r_pinky_joint',
}

# Mimic joints: (master_joint_name, multiplier, offset)
# PyBullet ignores <mimic> tags so we drive these manually.
MIMIC_JOINTS = {
    'r_thumb1_joint':  ('r_thumb_joint',  0.75, 0),
    'r_thumb3_joint':  ('r_thumb_joint',  1.0,  0),
    'r_index1_joint':  ('r_index_joint',  1.0,  0),
    'r_index3_joint':  ('r_index_joint',  1.0,  0),
    'r_middle1_joint': ('r_middle_joint', 1.0,  0),
    'r_middle3_joint': ('r_middle_joint', 1.0,  0),
    'r_ring1_joint':   ('r_ring_joint',  -0.1,  0),
    'r_ring3_joint':   ('r_ring_joint',   1.0,  0),
    'r_ring4_joint':   ('r_ring_joint',   1.0,  0),
    'r_pinky1_joint':  ('r_pinky_joint', -0.1,  0),
    'r_pinky3_joint':  ('r_pinky_joint',  1.0,  0),
    'r_pinky4_joint':  ('r_pinky_joint',  1.0,  0),
}

# Joint angle limits (radians) from config.yaml
JOINT_LIMITS = {
    'r_thumb_joint':  (-0.0873, 1.1345),
    'r_index_joint':  (-0.0873, 1.4835),
    'r_middle_joint': (-0.0873, 1.4835),
    'r_ring_joint':   (-0.0873, 1.4835),
    'r_pinky_joint':  (-0.0873, 1.4835),
}

# Smoothing factor for exponential moving average (0 = no smoothing, 1 = no update)
EMA_ALPHA = 0.4


def landmark_to_vec(landmark):
    """Convert a MediaPipe landmark to a numpy 3D vector."""
    return np.array([landmark.x, landmark.y, landmark.z])


def angle_between(a, b, c):
    """Angle at point b formed by vectors b→a and b→c, in radians."""
    v1 = a - b
    v2 = c - b
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return math.acos(np.clip(cos_angle, -1.0, 1.0))


def compute_finger_flex(landmarks):
    """
    Compute flex value [0, 1] for each finger from MediaPipe landmarks.
    0 = fully open, 1 = fully closed.

    Uses the angle at the PIP joint (middle joint of the finger).
    Straight finger ≈ π radians → flex = 0
    Fully curled ≈ 0.5 radians → flex = 1
    """
    flex = {}
    for finger, indices in FINGER_LANDMARKS.items():
        # Use the 3 middle landmarks to get the bend at the PIP/IP joint
        a = landmark_to_vec(landmarks.landmark[indices[0]])
        b = landmark_to_vec(landmarks.landmark[indices[1]])
        c = landmark_to_vec(landmarks.landmark[indices[2]])
        d = landmark_to_vec(landmarks.landmark[indices[3]])

        # Average of two joint angles for more robust measurement
        angle1 = angle_between(a, b, c)  # MCP angle
        angle2 = angle_between(b, c, d)  # PIP/DIP angle

        # Straight ≈ π, curled ≈ π/3
        avg_angle = (angle1 + angle2) / 2.0

        # Map: π → 0 (open), π/3 → 1 (closed)
        min_angle = math.pi / 3   # fully curled
        max_angle = math.pi       # fully straight
        flex_val = 1.0 - (avg_angle - min_angle) / (max_angle - min_angle)
        flex[finger] = np.clip(flex_val, 0.0, 1.0)

    return flex


def main():
    # --- PyBullet setup ---
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(0)

    # Camera angle: looking at the hand from the front
    p.resetDebugVisualizerCamera(
        cameraDistance=0.35,
        cameraYaw=180,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, -0.1]
    )
    # Hide UI clutter
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # Load URDF
    urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "right_hand.urdf")
    hand_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)

    # Build joint name → index mapping
    joint_map = {}
    for i in range(p.getNumJoints(hand_id)):
        info = p.getJointInfo(hand_id, i)
        joint_map[info[1].decode('utf-8')] = i

    print("Loaded joints:", list(joint_map.keys()))

    # --- MediaPipe setup ---
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    # Smoothed flex values
    smooth_flex = {f: 0.0 for f in FINGER_LANDMARKS}

    print("Hand tracking started. Show your right hand to the camera. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally so it acts like a mirror
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks on webcam feed
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Compute raw flex
            raw_flex = compute_finger_flex(hand_landmarks)

            # Apply EMA smoothing
            for finger in smooth_flex:
                smooth_flex[finger] = (
                    EMA_ALPHA * raw_flex[finger]
                    + (1 - EMA_ALPHA) * smooth_flex[finger]
                )

            # Drive the 5 control joints
            for finger, joint_name in CONTROL_JOINTS.items():
                if joint_name not in joint_map:
                    continue
                lower, upper = JOINT_LIMITS[joint_name]
                target = lower + smooth_flex[finger] * (upper - lower)
                p.setJointMotorControl2(
                    hand_id, joint_map[joint_name],
                    p.POSITION_CONTROL, targetPosition=target,
                    force=1000, maxVelocity=5.0,
                )

            # Drive mimic joints
            for mimic_name, (master_name, mult, offset) in MIMIC_JOINTS.items():
                if mimic_name not in joint_map or master_name not in joint_map:
                    continue
                master_pos = p.getJointState(hand_id, joint_map[master_name])[0]
                target = mult * master_pos + offset
                p.setJointMotorControl2(
                    hand_id, joint_map[mimic_name],
                    p.POSITION_CONTROL, targetPosition=target,
                    force=1000, maxVelocity=5.0,
                )

            # Show flex values on webcam feed
            y = 30
            for finger in FINGER_LANDMARKS:
                txt = f"{finger}: {smooth_flex[finger]:.2f}"
                cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25

        p.stepSimulation()

        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    p.disconnect()


if __name__ == '__main__':
    main()
