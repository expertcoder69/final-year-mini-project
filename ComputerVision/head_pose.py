import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# Head tracking settings
# -----------------------------

center_yaw = 0.0
center_pitch = 0.0

calibrated = False

smooth_yaw = 0.0
smooth_pitch = 0.0

SMOOTHING = 0.2
DEAD_ZONE = 3.0


# -----------------------------
# MediaPipe setup
# -----------------------------

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


# -----------------------------
# Webcam
# -----------------------------

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect face
    result = detector.detect(mp_image)


    # -----------------------------
    # Head pose calculation
    # -----------------------------

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        h, w, _ = frame.shape

        # Important face points
        nose = landmarks[1]
        chin = landmarks[152]
        # Anatomical left/right
        left_eye = landmarks[263]
        right_eye = landmarks[33]

        left_mouth = landmarks[291]
        right_mouth = landmarks[61]


        # -----------------------------
        # 2D image points
        # -----------------------------

        image_points = np.array([
            [nose.x * w, nose.y * h],
            [chin.x * w, chin.y * h],
            [left_eye.x * w, left_eye.y * h],
            [right_eye.x * w, right_eye.y * h],
            [left_mouth.x * w, left_mouth.y * h],
            [right_mouth.x * w, right_mouth.y * h]
        ], dtype=np.float64)


        # -----------------------------
        # Approximate 3D face model
        # -----------------------------

        model_points = np.array([
            [0.0, 0.0, 0.0],           # Nose
            [0.0, -330.0, -65.0],      # Chin
            [-225.0, 170.0, -135.0],   # Left eye
            [225.0, 170.0, -135.0],    # Right eye
            [-150.0, -150.0, -125.0],  # Left mouth
            [150.0, -150.0, -125.0]    # Right mouth
        ], dtype=np.float64)


        # -----------------------------
        # Camera matrix
        # -----------------------------

        focal_length = w

        center = (
            w / 2,
            h / 2
        )

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))


        # -----------------------------
        # Solve head pose
        # -----------------------------

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
            roll = angles[2]

            # -----------------------------
            # Calibration
            # -----------------------------

            if calibrated:

                relative_yaw = yaw - center_yaw
                relative_pitch = pitch - center_pitch

            else:

                relative_yaw = 0.0
                relative_pitch = 0.0

            # -----------------------------
            # Smoothing
            # -----------------------------

            smooth_yaw = (
                (1 - SMOOTHING) * smooth_yaw
                + SMOOTHING * relative_yaw
            )

            smooth_pitch = (
                (1 - SMOOTHING) * smooth_pitch
                + SMOOTHING * relative_pitch
            )

            # -----------------------------
            # Dead zone
            # -----------------------------

            if abs(smooth_yaw) < DEAD_ZONE:
                smooth_yaw = 0

            if abs(smooth_pitch) < DEAD_ZONE:
                smooth_pitch = 0

            cv2.putText(
                frame,
                f"Yaw: {smooth_yaw:.1f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Pitch: {smooth_pitch:.1f}",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Press C to calibrate",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    # -----------------------------
    # Limit head movement
    # -----------------------------

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


    # -----------------------------
    # Display webcam
    # -----------------------------

    cv2.imshow(
        "Dark Hunt - Head Pose",
        frame
    )


    # Press Q to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):

        center_yaw = yaw
        center_pitch = pitch

        smooth_yaw = 0.0
        smooth_pitch = 0.0

        calibrated = True

        print("Calibration complete!")
        print(f"Center Yaw: {center_yaw:.2f}")
        print(f"Center Pitch: {center_pitch:.2f}")


    if key == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------

cap.release()
detector.close()
cv2.destroyAllWindows()