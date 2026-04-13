import cv2
import torch
import numpy as np
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

# MPS has a tensor contiguity bug with ViT conv2d in PyTorch 2.5
# Force CPU for now — still usable for offline ground truth extraction
device = torch.device("cpu")
print(f"Using device: {device}")

# Load pipeline (auto-downloads model weights on first run)
print("Loading WiLoR model...")
pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float32, verbose=False)
print("Model loaded!")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press ESC to quit")

# Joint connections for drawing skeleton
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20) # pinky
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pipe.predict(frame_rgb)

    for det in results:
        # Draw bounding box
        bbox = det["hand_bbox"]
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        if "wilor_preds" in det:
            preds = det["wilor_preds"]
            # Draw 2D keypoints
            kpts = preds["pred_keypoints_2d"][0]  # (21, 2)
            for j in range(kpts.shape[0]):
                x, y = int(kpts[j, 0]), int(kpts[j, 1])
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            # Draw skeleton
            for a, b in CONNECTIONS:
                x1, y1 = int(kpts[a, 0]), int(kpts[a, 1])
                x2, y2 = int(kpts[b, 0]), int(kpts[b, 1])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 200), 2)

    cv2.imshow("WiLoR Real-time", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
