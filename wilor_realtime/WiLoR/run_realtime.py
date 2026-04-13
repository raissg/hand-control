import torch
import cv2
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import cam_crop_to_full
from ultralytics import YOLO

# Use MPS on Apple Silicon, fallback to CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load models
print("Loading WiLoR model...")
model, model_cfg = load_wilor(
    checkpoint_path='./pretrained_models/wilor_final.ckpt',
    cfg_path='./pretrained_models/model_config.yaml'
)
model = model.to(device)
model.eval()

print("Loading hand detector...")
detector = YOLO('./pretrained_models/detector.pt')
detector.to(device)

print("Ready! Press ESC to quit.")

# Joint connections for skeleton drawing
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

def project_full_img(points, cam_trans, focal_length, img_res):
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3)
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = camera_center[0]
    K[1,2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:]
    V_2d = (K @ points.T).T
    return V_2d[..., :-1]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()

    # Detect hands
    detections = detector(frame, conf=0.3, verbose=False)[0]
    bboxes = []
    is_right = []
    for det in detections:
        bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(bbox[:4].tolist())

    if len(bboxes) > 0:
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        dataset = ViTDetDataset(model_cfg, frame, boxes, right, rescale_factor=2.0)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                ir = batch['right'][n].cpu().numpy()
                verts[:, 0] = (2 * ir - 1) * verts[:, 0]
                joints[:, 0] = (2 * ir - 1) * joints[:, 0]
                cam_t = pred_cam_t_full[n]

                # Project 3D joints to 2D
                kpts_2d = project_full_img(joints, cam_t, scaled_focal_length.cpu().numpy(), img_size[n].cpu().numpy())

                # Draw joints
                for j in range(kpts_2d.shape[0]):
                    x, y = int(kpts_2d[j, 0]), int(kpts_2d[j, 1])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                # Draw skeleton
                for a, b in CONNECTIONS:
                    x1, y1 = int(kpts_2d[a, 0]), int(kpts_2d[a, 1])
                    x2, y2 = int(kpts_2d[b, 0]), int(kpts_2d[b, 1])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 200), 2)

    # Show FPS
    fps = 1.0 / (time.time() - t0 + 1e-6)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("WiLoR Real-time", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
