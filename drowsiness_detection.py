import cv2
import mediapipe as mp
from scipy.spatial import distance

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 20

counter = 0


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    return (A + B) / (2.0 * C)


while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not working")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            h, w, _ = frame.shape

            left_eye = []
            right_eye = []

           
            for i in LEFT_EYE:
                x = int(face.landmark[i].x * w)
                y = int(face.landmark[i].y * h)
                left_eye.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

           
            for i in RIGHT_EYE:
                x = int(face.landmark[i].x * w)
                y = int(face.landmark[i].y * h)
                right_eye.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            ear = (left_ear + right_ear) / 2.0

            
            cv2.putText(frame,
                        f"EAR: {ear:.2f}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2)

            
            if ear < EAR_THRESHOLD:
                counter += 1

                if counter >= FRAME_THRESHOLD:
                    cv2.putText(frame,
                                "DROWSY ALERT!",
                                (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0, 0, 255),
                                3)
            else:
                counter = 0

    cv2.imshow("Drowsiness Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
