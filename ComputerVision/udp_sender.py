import cv2
import mediapipe as mp
import numpy as np
import socket

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# UDP SETUP
# =========================

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# =========================
# MEDIAPIPE SETUP
# =========================

model_path = "face_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.FaceLandmarker.create_from_options(options)


# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

print("✅ Webcam connected!")
print("Press C to calibrate")
print("Press Q to quit")


# =========================
# CALIBRATION
# =========================

center_yaw = 0.0
center_pitch = 0.0
calibrated = False

smooth_yaw = 0.0
smooth_pitch = 0.0

SMOOTHING = 0.2
DEAD_ZONE = 3.0


# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read webcam")
        break

    h, w, _ = frame.shape

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(mp_image)

    # =========================
    # FACE DETECTED
    # =========================

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        # Important landmarks
        nose = landmarks[1]
        chin = landmarks[152]

        left_eye = landmarks[263]
        right_eye = landmarks[33]

        left_mouth = landmarks[291]
        right_mouth = landmarks[61]

        # Image coordinates
        image_points = np.array([
            [nose.x * w, nose.y * h],
            [chin.x * w, chin.y * h],
            [left_eye.x * w, left_eye.y * h],
            [right_eye.x * w, right_eye.y * h],
            [left_mouth.x * w, left_mouth.y * h],
            [right_mouth.x * w, right_mouth.y * h]
        ], dtype=np.float64)

        # 3D face model
        model_points = np.array([
            [0.0, 0.0, 0.0],
            [0.0, -330.0, -65.0],
            [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0, -150.0, -125.0]
        ], dtype=np.float64)

        # Camera matrix
        focal_length = w
        center = (w / 2, h / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        # Solve head pose
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            angles, _, _, _, _, _ = cv2.RQDecomp3x3(
                rotation_matrix
            )

            pitch = angles[0]
            yaw = angles[1]

            # =========================
            # CALIBRATION
            # =========================

            if calibrated:

                relative_yaw = yaw - center_yaw
                relative_pitch = pitch - center_pitch

            else:

                relative_yaw = 0.0
                relative_pitch = 0.0

            # =========================
            # SMOOTHING
            # =========================

            smooth_yaw = (
                (1 - SMOOTHING) * smooth_yaw
                + SMOOTHING * relative_yaw
            )

            smooth_pitch = (
                (1 - SMOOTHING) * smooth_pitch
                + SMOOTHING * relative_pitch
            )

            # =========================
            # DEAD ZONE
            # =========================

            if abs(smooth_yaw) < DEAD_ZONE:
                smooth_yaw = 0

            if abs(smooth_pitch) < DEAD_ZONE:
                smooth_pitch = 0

            # =========================
            # LIMIT VALUES
            # =========================

            smooth_yaw = np.clip(
                smooth_yaw,
                -70,
                70
            )

            smooth_pitch = np.clip(
                smooth_pitch,
                -35,
                35
            )

            # =========================
            # SEND UDP
            # =========================

            message = f"{smooth_yaw:.2f},{smooth_pitch:.2f}"

            sock.sendto(
                message.encode("utf-8"),
                (UDP_IP, UDP_PORT)
            )

            # Display values
            cv2.putText(
                frame,
                f"Yaw: {smooth_yaw:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Pitch: {smooth_pitch:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            if calibrated:
                cv2.putText(
                    frame,
                    "CALIBRATED",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "Press C to calibrate",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

    # Show webcam window
    cv2.imshow("Dark Hunt - Head Tracking", frame)

    # Keyboard input
    key = cv2.waitKey(1) & 0xFF

    # =========================
    # KEYBOARD
    # =========================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):

        if result.face_landmarks:

            center_yaw = yaw
            center_pitch = pitch

            smooth_yaw = 0.0
            smooth_pitch = 0.0

            calibrated = True

            print("\n✅ Calibration complete!")
            print(f"Center Yaw: {center_yaw:.2f}")
            print(f"Center Pitch: {center_pitch:.2f}")

    if key == ord("q"):
        break


# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()
sock.close()

print("UDP sender stopped.")