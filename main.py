import cv2
import torch
import numpy as np
import mediapipe as mp
from scipy.interpolate import RectBivariateSpline
import argparse

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

alpha = 0.2
previous_depth = 0.0
depth_scale = 1.0

def apply_ema_filter(current_depth):
    global previous_depth
    filtered_depth = alpha * current_depth + (1 - alpha) * previous_depth
    previous_depth = filtered_depth
    return filtered_depth

def depth2distance(depth_value, depth_scale):
    return 1.0 / (depth_value * depth_scale)

def depth2distance1(depth_value, depth_scale):
    return -1.0 / (depth_value * depth_scale)

# Initialize argument parser
parser = argparse.ArgumentParser(description="Pose estimation and depth estimation from video")
parser.add_argument("--input", required=True, help="Path to input video file")
parser.add_argument("--output", required=True, help="Path to output video file")
args = parser.parse_args()

# Open the input video file
cap = cv2.VideoCapture(args.input)

# Initialize the output video writer
output_video = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'MP4V'), 30,
                               (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(img)

    if results.pose_landmarks is not None:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))

        waist_landmarks = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]]

        mid_point = ((waist_landmarks[0].x + waist_landmarks[1].x) / 2,
                     (waist_landmarks[0].y + waist_landmarks[1].y) / 2)
        mid_x, mid_y = mid_point

        imgbatch = transform(img).to('cpu')

        with torch.no_grad():
            prediction = midas(imgbatch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()

        output = prediction.cpu().numpy()
        output_norm = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        h, w = output_norm.shape
        x_grid = np.arange(w)
        y_grid = np.arange(h)

        spline = RectBivariateSpline(y_grid, x_grid, output_norm)
        depth_mid_filt = spline(mid_y, mid_x)
        depth_midas = depth2distance(depth_mid_filt, depth_scale)
        depth_mid_filt = (apply_ema_filter(depth_midas) / 10)[0][0]

        # Convert depth map to colored image
        depth_colored = cv2.applyColorMap((output_norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

        # Draw mediapoint landmarks on the image
        for landmark in results.pose_landmarks.landmark:
            x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
            cv2.circle(depth_colored, (x, y), 5, (0, 255, 0), -1)

        # Draw lines between landmark points
        connections = mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_point = connection[0]
            end_point = connection[1]
            start_pos = (int(results.pose_landmarks.landmark[start_point].x * img.shape[1]),
                         int(results.pose_landmarks.landmark[start_point].y * img.shape[0]))
            end_pos = (int(results.pose_landmarks.landmark[end_point].x * img.shape[1]),
                       int(results.pose_landmarks.landmark[end_point].y * img.shape[0]))
            cv2.line(depth_colored, start_pos, end_pos, (255, 0, 0), 3)

        # Overlay depth map on top of the image
        merged_output = cv2.addWeighted(depth_colored, 0.5, img, 0.5, 0)

        cv2.putText(merged_output,
                    f'Depth in unit: {np.format_float_positional(depth_mid_filt, precision=3)} (FPS: {fps})',
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 3)

        cv2.imshow('Video Feed', merged_output)
        
        # Write the frame to the output video
        output_video.write(cv2.cvtColor(merged_output, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Release the output video writer
output_video.release()
