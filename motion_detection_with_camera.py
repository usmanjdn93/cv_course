import numpy as np
import cv2
import time

def draw_flow(img, floq, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = floq[y, x].T

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (x2, y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi

    v = np.sqrt(fx**2 + fy**2)
    hsv = np.zeros((h, w, 3), np.uint8)

    hsv[..., 0] = (ang * (180 / np.pi / 2)).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera is accessible
if not cap.isOpened():
    print("error: Unable to access the camera.")
    exit()

# Allow the camera to initialize
time.sleep(2)

# Read the first frame
suc, prev = cap.read()
if not suc:
    print("error: initial frame not captured.")
    cap.release()
    exit()

# Convert the initial frame to grayscale
pregray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:
    suc, img = cap.read()
    if not suc:
        print("error: frame not captured")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    start = time.time()

    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(pregray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    pregray = gray

    end = time.time()
    fps = 1 / (end - start)
    print(f'FPS: {fps:.2f}')

    # Display the results
    cv2.imshow('Flow Visualization', draw_flow(gray, flow))
    cv2.imshow('Flow HSV', draw_hsv(flow))

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
