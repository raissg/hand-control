import os
import sys
import time
import types
import urllib.request
import cv2
import numpy as np

## ------------------- ##
## Inverse Kinematics  ##
## ------------------- ##
from keypoint2mano import Keypoints2Mano
from inverse_kinematics.armatures import *
from inverse_kinematics.models import *
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
k2m = Keypoints2Mano(model_path=os.path.join(_SCRIPT_DIR, 'MANO_RIGHT.npz'))

## ------------------- ##
##      Visualizer     ##
## ------------------- ##
# Open3D pulls in Dash; Dash calls comm.create_comm() at import time. Conda ships
# `comm` without a Jupyter kernel, so that raises NotImplementedError in plain
# `python main.py`. A no-op shim avoids loading the real comm until needed.
if "comm" not in sys.modules:
    _comm_stub = types.ModuleType("comm")

    class _NoOpComm:
        def send(self, *args, **kwargs):
            pass

        def on_msg(self, callback):
            return callback

    _comm_stub.create_comm = lambda *a, **k: _NoOpComm()
    _comm_stub.get_comm_manager = lambda: None
    sys.modules["comm"] = _comm_stub

import open3d as o3d

mesh = o3d.geometry.TriangleMesh()
verts = k2m.get_vertices()
faces = k2m.get_faces()
mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.9, 0.8, 0.7])

vis = o3d.visualization.Visualizer()
vis.create_window(window_name='EtherPose Viewer')
vis.add_geometry(mesh)

## ------------------- ##
##      Mediapipe      ##
## ------------------- ##
import mediapipe as mp
from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe.tasks.python.vision import RunningMode, drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import hand_landmarker

_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_model_path = os.path.join(_SCRIPT_DIR, "models", "hand_landmarker.task")
if not os.path.isfile(_model_path):
    os.makedirs(os.path.dirname(_model_path), exist_ok=True)
    urllib.request.urlretrieve(_HAND_MODEL_URL, _model_path)

_hand_options = hand_landmarker.HandLandmarkerOptions(
    base_options=mp_base_options.BaseOptions(model_asset_path=_model_path),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hands = hand_landmarker.HandLandmarker.create_from_options(_hand_options)
_hand_connections = hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS

# Open video stream (0 for the default camera, or replace with video file path)
cap = cv2.VideoCapture(0)
_t0 = time.monotonic()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    timestamp_ms = int((time.monotonic() - _t0) * 1000)
    results = hands.detect_for_video(mp_image, timestamp_ms)

    if results.hand_world_landmarks:
        for world in results.hand_world_landmarks:
            hand_landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in world])
            k2m.get_mano_params(hand_landmarks_array)
            verts = k2m.get_vertices()
            faces = k2m.get_faces()

            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            vis.update_geometry(mesh)

        for norm in results.hand_landmarks:
            mp_drawing.draw_landmarks(frame, norm, _hand_connections)
    vis.poll_events()
    vis.update_renderer()

    cv2.imshow('MediaPipe Hands', frame)

    if cv2.waitKey(5) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
