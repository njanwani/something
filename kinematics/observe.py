# from ultralytics import settings
# print(settings)
# quit()
from ultralytics import YOLO
import cv2
import numpy as np


class PoseObserver2D:
    
    def __init__(self, norm_length):  
        self.left_arm_pxls = np.zeros(3)
        self.right_arm_pxls = np.zeros(3)      
        self.norm_length = norm_length
        self.model = YOLO("yolov8l-pose.pt")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            exit()
    
    def read_pose(self):
        ret, frame = self.cap.read()
        results = self.model(frame, verbose=False)
        # Get keypoints (if any person detected)
        keypoints = results[0].keypoints
        if keypoints is not None and len(keypoints.xy) > 0:
            # Get first person (index 0)
            person_kpts = keypoints.xy[0].cpu().numpy()  # shape (17, 2)

            # Arm keypoints (indices)
            left_arm_idxs = [5, 7, 9]
            right_arm_idxs = [6, 8, 10]

            self.left_arm_pxls  = np.array(person_kpts[left_arm_idxs])
            self.right_arm_pxls = np.array(person_kpts[right_arm_idxs])
            
    def arm_length(self, arm_arr):
        return np.sum(np.linalg.norm(np.diff(arm_arr, axis=0)))
            
    def arm(self, arm_arr):
        flipped = ((arm_arr - arm_arr[0]) / self.arm_length(arm_arr)) * self.norm_length
        flipz   = np.array([[1, -1]])
        return flipped * flipz
    
    @property
    def left_arm(self):
        return self.arm(self.left_arm_pxls)
    
    @property
    def right_arm(self):
        return self.arm(self.right_arm_pxls)


if __name__ == '__main__':
    
    # Load YOLOv8 pose model
    model = YOLO("yolov8l-pose.pt")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Run pose estimation
        results = model(frame, verbose=False)

        # Draw annotated results
        annotated_frame = results[0].plot()

        # Get keypoints (if any person detected)
        keypoints = results[0].keypoints
        if keypoints is not None and len(keypoints.xy) > 0:
            # Get first person (index 0)
            person_kpts = keypoints.xy[0].cpu().numpy()  # shape (17, 2)

            # Arm keypoints (indices)
            left_arm_idxs = [5, 7, 9]
            right_arm_idxs = [6, 8, 10]

            left_arm = person_kpts[left_arm_idxs]
            right_arm = person_kpts[right_arm_idxs]

            # Draw arm points on frame (optional)
            for (x, y) in np.vstack([left_arm, right_arm]):
                cv2.circle(annotated_frame, (int(x), int(y)), 15, (0, 255, 255), -1)

            # Print coordinates (optional)
            print("Left Arm:", left_arm)
            print("Right Arm:", right_arm)

        cv2.imshow("YOLOv8 Pose Estimation (Arms Highlighted)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
