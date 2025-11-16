import cv2
import mediapipe as mp
import numpy as np
# Setup MediaPipe
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def lm_to_np(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

def main():
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for MediaPipe
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run model
            results = pose.process(img_rgb)

            # Draw 2D skeleton if detected
            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(0,0,255), thickness=2),
                )

                lm = results.pose_landmarks.landmark

                right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_elbow    = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_wrist    = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

                # print("Right Shoulder:", right_shoulder.x, right_shoulder.y, right_shoulder.z)
                # print("Right Elbow:", right_elbow.x, right_elbow.y, right_elbow.z)
                # print("Right Wrist:", right_wrist.x, right_wrist.y, right_wrist.z)
                arm = np.round(lm_to_np(right_wrist) - lm_to_np(right_shoulder), 3)
                print(np.round(lm_to_np(right_wrist) - lm_to_np(right_shoulder), 3))
                for i, val in enumerate(arm):
                    cv2.putText(
                        frame,
                        f"{val:.3f}",
                        (200, 200 + i*200),   # position on the frame
                        cv2.FONT_HERSHEY_SIMPLEX,
                        8,
                        (0, 0, 0),
                        20,
                    )

            cv2.imshow("Pose", frame)
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
