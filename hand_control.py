"""
Real-time hand tracking → InMoov robot hand animation.
Uses vedo for 3D rendering (native arm64 / Apple Silicon M-series compatible).

Pipeline:
  Webcam → MediaPipe (21 landmarks) → finger flex angles
        → Forward kinematics → vedo 3D window

Usage:
    pip install -r requirements.txt
    python hand_control.py

Press 'q' in the OpenCV window to quit. The window shows the raw camera feed
(left, same RGB passed to MediaPipe) and the tracked overlay (right).
"""

import math
import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from vedo import Plotter, Mesh

# ─── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# MediaPipe: 4 landmark indices per finger (base → tip)
FINGER_LANDMARKS = {
    'thumb':  [1, 2, 3, 4],
    'index':  [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring':   [13, 14, 15, 16],
    'pinky':  [17, 18, 19, 20],
}

# Joint limits in radians (from InMoov config.yaml, converted from degrees)
JOINT_LIMITS = {
    'thumb':  (-0.0873, 1.1345),   # -5° to 65°
    'index':  (-0.0873, 1.4835),   # -5° to 85°
    'middle': (-0.0873, 1.4835),
    'ring':   (-0.0873, 1.4835),
    'pinky':  (-0.0873, 1.4835),
}

EMA_ALPHA = 0.4  # smoothing (0 = no smoothing, 1 = frozen)


def _draw_panel_caption(img, x0, text):
    """Small bar + white text at top-left of a horizontal region starting at x0."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.55, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    y = 26
    cv2.rectangle(img, (x0 + 4, y - th - 6), (x0 + tw + 12, y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x0 + 8, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

# ─── Kinematic tree ───────────────────────────────────────────────────────────
# Each entry defines one joint in the URDF chain (expanded from xacro, side=r, flip=1).
#
# finger: which finger drives this joint
# mult:   multiplier on the control angle (mimic ratio from URDF)
#         e.g. thumb1 mimics thumb at 0.75x, ring1 mimics ring at -0.1x (splay)

JOINTS = [
    # Palm fixed to world
    dict(parent='base',         child='r_hand_link',    finger=None,
         xyz=(0, 0, 0.05),              rpy=(0, 0, 1.57),              axis='z', mult=0),

    # ── Thumb ──
    dict(parent='r_hand_link',  child='r_thumb1_link',  finger='thumb',
         xyz=(0, 0.029, -0.0577),    rpy=(0.1, 0, 0),            axis='z', mult=0.75),
    dict(parent='r_thumb1_link', child='r_thumb2_link', finger='thumb',
         xyz=(-0.00052, 0.02725, -0.013), rpy=(0.825, -0.1, 0.3), axis='y', mult=1.0),
    dict(parent='r_thumb2_link', child='r_thumb3_link', finger='thumb',
         xyz=(0, 0, -0.035),         rpy=(0, 0, 0),              axis='y', mult=1.0),

    # ── Index ──
    dict(parent='r_hand_link',  child='r_index1_link',  finger='index',
         xyz=(-0.0015, 0.0342, -0.119), rpy=(0.1, 0, 0),         axis='y', mult=1.0),
    dict(parent='r_index1_link', child='r_index2_link', finger='index',
         xyz=(0, 0.001, -0.03595),   rpy=(0, 0, 0),              axis='y', mult=1.0),
    dict(parent='r_index2_link', child='r_index3_link', finger='index',
         xyz=(0, 0, -0.024),         rpy=(0, 0, 0),              axis='y', mult=1.0),

    # ── Middle ──
    dict(parent='r_hand_link',  child='r_middle1_link', finger='middle',
         xyz=(-0.00175, 0.007, -0.12325), rpy=(0, 0, 0),         axis='y', mult=1.0),
    dict(parent='r_middle1_link', child='r_middle2_link', finger='middle',
         xyz=(0, 0, -0.0389),        rpy=(0, 0, 0),              axis='y', mult=1.0),
    dict(parent='r_middle2_link', child='r_middle3_link', finger='middle',
         xyz=(0, 0.0005, -0.0259),   rpy=(0, 0, 0),              axis='y', mult=1.0),

    # ── Ring ──
    dict(parent='r_hand_link',  child='r_ring1_link',   finger='ring',
         xyz=(0, -0.00705, -0.0794), rpy=(0.7, 0, 0),            axis='z', mult=-0.1),
    dict(parent='r_ring1_link', child='r_ring2_link',   finger='ring',
         xyz=(0.00126, -0.0351, -0.0166), rpy=(-0.7775, 0, 0),   axis='y', mult=1.0),
    dict(parent='r_ring2_link', child='r_ring3_link',   finger='ring',
         xyz=(0, 0.0005, -0.0345),   rpy=(0, 0, 0),              axis='y', mult=1.0),
    dict(parent='r_ring3_link', child='r_ring4_link',   finger='ring',
         xyz=(0, 0.0004, -0.0229),   rpy=(0, 0, 0),              axis='y', mult=1.0),

    # ── Pinky ──
    dict(parent='r_hand_link',  child='r_pinky1_link',  finger='pinky',
         xyz=(0, -0.0270, -0.0555),  rpy=(0.7, 0, 0),            axis='z', mult=-0.1),
    dict(parent='r_pinky1_link', child='r_pinky2_link', finger='pinky',
         xyz=(0, -0.046, -0.0228),   rpy=(-0.93, 0, 0),          axis='y', mult=1.0),
    dict(parent='r_pinky2_link', child='r_pinky3_link', finger='pinky',
         xyz=(0, 0.0005, -0.031),    rpy=(0, 0, 0),              axis='y', mult=1.0),
    dict(parent='r_pinky3_link', child='r_pinky4_link', finger='pinky',
         xyz=(0, 0.0004, -0.0208),   rpy=(0, 0, 0),              axis='y', mult=1.0),
]

# Link name → STL file
LINK_MESHES = {
    'r_hand_link':    'meshes/r_hand.stl',
    'r_thumb1_link':  'meshes/r_thumb5_1.stl',
    'r_thumb2_link':  'meshes/thumb5_2.stl',
    'r_thumb3_link':  'meshes/thumb5_3.stl',
    'r_index1_link':  'meshes/index3_1.stl',
    'r_index2_link':  'meshes/index3_2.stl',
    'r_index3_link':  'meshes/index3_3.stl',
    'r_middle1_link': 'meshes/middle3_1.stl',
    'r_middle2_link': 'meshes/middle3_2.stl',
    'r_middle3_link': 'meshes/middle3_3.stl',
    'r_ring1_link':   'meshes/r_ring3_1.stl',
    'r_ring2_link':   'meshes/ring3_2.stl',
    'r_ring3_link':   'meshes/ring3_3.stl',
    'r_ring4_link':   'meshes/ring3_4.stl',
    'r_pinky1_link':  'meshes/r_pinky3_1.stl',
    'r_pinky2_link':  'meshes/pinky3_2.stl',
    'r_pinky3_link':  'meshes/pinky3_3.stl',
    'r_pinky4_link':  'meshes/pinky3_4.stl',
}

# ─── Transform math ───────────────────────────────────────────────────────────

def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=float)

def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=float)

def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)

def _trans(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T

def _rpy(r, p, y):
    """URDF RPY: extrinsic XYZ rotation = Rz @ Ry @ Rx"""
    return _rz(y) @ _ry(p) @ _rx(r)

def joint_origin_matrix(xyz, rpy):
    """4x4 transform for a joint's origin (translation + fixed orientation)."""
    T = _trans(*xyz)
    T[:3, :3] = _rpy(*rpy)[:3, :3]
    return T

def axis_rotation_matrix(axis, angle):
    """4x4 rotation matrix around x/y/z by angle radians."""
    return {'x': _rx, 'y': _ry, 'z': _rz}[axis](angle)

def transform_points(pts, T):
    """Apply 4x4 matrix T to Nx3 point array."""
    ones = np.ones((len(pts), 1))
    return (T @ np.hstack([pts, ones]).T).T[:, :3]

# ─── Forward kinematics ───────────────────────────────────────────────────────

def compute_world_transforms(finger_angles):
    """
    Walk the kinematic tree and compute each link's 4x4 world transform.
    finger_angles: dict {finger_name: angle_in_radians}
    """
    world_T = {'base': np.eye(4)}

    for j in JOINTS:
        parent = j['parent']
        child  = j['child']
        finger = j['finger']
        mult   = j['mult']

        # Get joint angle
        if finger is None:
            angle = 0.0  # fixed joint (palm to world)
        else:
            angle = finger_angles[finger] * mult

        T_origin = joint_origin_matrix(j['xyz'], j['rpy'])
        T_rot    = axis_rotation_matrix(j['axis'], angle)

        world_T[child] = world_T[parent] @ T_origin @ T_rot

    return world_T

# ─── Finger flex from MediaPipe landmarks ─────────────────────────────────────

def landmark_vec(lm, idx):
    l = lm[idx]
    return np.array([l.x, l.y, l.z])

def angle_at(a, b, c):
    """Angle at point b (radians)."""
    v1 = a - b
    v2 = c - b
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return math.acos(np.clip(cos_a, -1.0, 1.0))

def compute_finger_flex(landmarks):
    """
    Returns dict {finger: flex_value} where 0.0 = open, 1.0 = closed.
    Averages angles at the MCP and PIP joints of each finger.
    """
    flex = {}
    for finger, idx in FINGER_LANDMARKS.items():
        a = landmark_vec(landmarks, idx[0])
        b = landmark_vec(landmarks, idx[1])
        c = landmark_vec(landmarks, idx[2])
        d = landmark_vec(landmarks, idx[3])
        ang = (angle_at(a, b, c) + angle_at(b, c, d)) / 2.0
        # π = straight (open), π/3 = fully curled (closed)
        flex[finger] = np.clip(1.0 - (ang - math.pi/3) / (math.pi - math.pi/3), 0.0, 1.0)
    return flex

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.chdir(SCRIPT_DIR)

    # Load original mesh vertices (STL, in mm → scale to meters)
    print("Loading meshes...")
    original_pts = {}
    meshes = {}
    for link, stl in LINK_MESHES.items():
        m = Mesh(stl)
        m.scale(0.001)  # mm → meters
        m.color('lightgray')
        original_pts[link] = m.points.copy()
        meshes[link] = m

    # Set up vedo plotter (non-interactive so we control the loop)
    plt = Plotter(title='InMoov Hand', size='fullscreen', interactive=False)
    plt.show(list(meshes.values()), resetcam=True, viewup='z')
    # Camera: look at palm, fingers pointing up
    plt.camera.SetPosition(0, -0.5, -0.06)
    plt.camera.SetFocalPoint(0, 0, -0.06)
    plt.camera.SetViewUp(0, 0, -1)

    # MediaPipe setup — download model if needed
    model_path = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')
    if not os.path.exists(model_path):
        print("Downloading MediaPipe hand landmarker model...")
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded.")

    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    # Connections for drawing (21 landmark pairs)
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    smooth_flex = {f: 0.0 for f in FINGER_LANDMARKS}
    frame_ts = 0
    print("Ready. Show your right hand. Press 'q' in the OpenCV window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        # Exact pixels sent to MediaPipe (before landmarks / HUD on this buffer)
        camera_input_bgr = frame.copy()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_ts += 33  # ~30fps timestamp in ms
        result = detector.detect_for_video(mp_image, frame_ts)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]

            # Draw landmarks
            pts_px = [(int(l.x * w), int(l.y * h)) for l in lm]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts_px[a], pts_px[b], (0, 200, 0), 2)
            for pt in pts_px:
                cv2.circle(frame, pt, 4, (0, 255, 0), -1)

            raw = compute_finger_flex(lm)
            for f in smooth_flex:
                smooth_flex[f] = EMA_ALPHA * raw[f] + (1 - EMA_ALPHA) * smooth_flex[f]

        # Map flex [0,1] → joint angle [lower, upper]
        finger_angles = {
            f: JOINT_LIMITS[f][0] + smooth_flex[f] * (JOINT_LIMITS[f][1] - JOINT_LIMITS[f][0])
            for f in FINGER_LANDMARKS
        }

        # Forward kinematics → world transforms
        world_T = compute_world_transforms(finger_angles)

        # Apply transforms to each mesh (in-place vertex update)
        for link, mesh in meshes.items():
            T = world_T[link]
            mesh.points = transform_points(original_pts[link], T)

        plt.render()

        # Overlay flex values on webcam
        y = 30
        for f, v in smooth_flex.items():
            cv2.putText(frame, f"{f}: {v:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25

        pair = np.hstack([camera_input_bgr, frame])
        _draw_panel_caption(pair, 0, "Camera (input to MediaPipe)")
        _draw_panel_caption(pair, w, "App view (landmarks + flex)")
        cv2.imshow('Hand Tracking', pair)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.close()

if __name__ == '__main__':
    main()
