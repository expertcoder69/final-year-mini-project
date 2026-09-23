import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open WebCam")
    exit()

print("WebCam Connected!!!")
print("Press 'Q' to Exit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read webCam")
        break

    cv2.imshow("Dark Hunt - WebCam test",frame)

    if cv2.waitKey(1) & 0xff == ord("q"):
        break

cap.release()
cv2.destroyAllWindows() 