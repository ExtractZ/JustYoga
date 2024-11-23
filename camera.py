import cv2
import mediapipe as mp


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
def getCamera():
    for n in range(6):
        cap = cv2.VideoCapture(n)
        if cap.isOpened():
            print(f"using camera index {n}")
            return cap
        cap.release()
    raise ValueError("none of the cameras work")
try:        
    cap = getCamera()
except ValueError as e:
    print(e)
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break


        frame = cv2.flip(frame, 1)


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        results = pose.process(rgb_frame)


        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Joints
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),  # Connections
            )


        cv2.imshow("Pose Tracking", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()