import cv2
import mediapipe as mp


from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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
# Webcam setup
# -----------------------------

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()


    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # OpenCV → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect face
    result = detector.detect(mp_image)


    # -----------------------------
    # Process face landmarks
    # -----------------------------

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        # Nose
        nose = landmarks[1]

        # Left eye
        left_eye = landmarks[33]

        # Right eye
        right_eye = landmarks[263]

        # Forehead
        forehead = landmarks[10]

        # Chin
        chin = landmarks[152]


        # Convert normalized coordinates → pixels

        h, w, _ = frame.shape

        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h)

        left_eye_x = int(left_eye.x * w)
        left_eye_y = int(left_eye.y * h)

        right_eye_x = int(right_eye.x * w)
        right_eye_y = int(right_eye.y * h)

        forehead_x = int(forehead.x * w)
        forehead_y = int(forehead.y * h)

        chin_x = int(chin.x * w)
        chin_y = int(chin.y * h)

        # -----------------------------
        # Calculate Yaw
        # -----------------------------

        eye_center_x = (left_eye.x + right_eye.x) / 2

        yaw = (nose.x - eye_center_x) * 100
        


        # -----------------------------
        # Draw important landmarks
        # -----------------------------

        cv2.circle(
            frame,
            (nose_x, nose_y),
            6,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            frame,
            (left_eye_x, left_eye_y),
            6,
            (255, 0, 0),
            -1
        )

        cv2.circle(
            frame,
            (right_eye_x, right_eye_y),
            6,
            (255, 0, 0),
            -1
        )

        cv2.circle(
            frame,
            (forehead_x, forehead_y),
            6,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            frame,
            (chin_x, chin_y),
            6,
            (0, 255, 0),
            -1
        )


        # Draw labels

        cv2.putText(
            frame,
            "NOSE",
            (nose_x + 10, nose_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            "LEFT EYE",
            (left_eye_x + 10, left_eye_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            "RIGHT EYE",
            (right_eye_x + 10, right_eye_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )


    # Show frame

    cv2.imshow(
        "Black Out - Head Tracking",
        frame
    )


    # Press Q to quit

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()

detector.close()

cv2.destroyAllWindows()