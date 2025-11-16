from ultralytics import YOLO
import cv2
import numpy as np
import torch


class PoseObserver2D:
    
    def __init__(self, norm_length, model='accurate', videofile=None):  
        self.left_arm_pxls = np.zeros(3)
        self.right_arm_pxls = np.zeros(3)      
        self.norm_length = norm_length
        # Choose model file
        model_path = "yolov8l-pose.pt" if model == 'accurate' else "yolov8n-pose.pt"

        # Detect best available device. On Apple Silicon prefer 'mps'. Otherwise prefer 'cuda' when available.
        device = 'cpu'
        try:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                # torch.backends.mps may not exist on older torch versions
                if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
                    device = 'mps'
        except Exception:
            # If any unexpected error happens while checking backends, fall back to cpu
            device = 'cpu'
        # Load the model (do not pass device to constructor because some ultralytics
        # versions do not accept a 'device' kwarg). After loading, try to move it to
        # the selected device using common interfaces and fall back to CPU on failure.
        try:
            self.model = YOLO(model_path)
            moved = False
            if device != 'cpu':
                try:
                    # Preferred: if model has a .to() method (nn.Module-like)
                    if hasattr(self.model, 'to'):
                        self.model.to(device)
                        moved = True
                    # ultralytics YOLO sometimes exposes a .model attribute
                    elif hasattr(self.model, 'model') and hasattr(self.model.model, 'to'):
                        self.model.model.to(device)
                        moved = True
                except Exception as e:
                    print(f"Warning: attempted to move model to '{device}' but failed: {e}")

            print(f"Loaded YOLO model '{model_path}' (requested device: {device}, moved: {moved})")
        except Exception as e:
            print(f"Error: failed to load YOLO model '{model_path}': {e}")
            raise
        if videofile is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open camera.")
                exit()
        else:
            self.cap = cv2.VideoCapture(videofile)
        
        self.frame = np.zeros((1,1))
    
    def read_pose(self):
        ret, self.frame = self.cap.read()
        results = self.model(self.frame, verbose=False)
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
        
        return ret
            
    def arm_length(self, arm_arr):
        return np.sum(np.linalg.norm(np.diff(arm_arr, axis=0)))
            
    def arm(self, arm_arr):
        flipped = ((arm_arr - arm_arr[0]) / self.arm_length(arm_arr)) * self.norm_length
        flipz   = np.array([[1, -1]])
        return flipped * flipz
    
    def show_keypoints(self):
        for pxl in self.left_arm_pxls:
            cv2.circle(
                self.frame,
                center=pxl.astype(int),
                radius=5,
                color=(0, 255, 0),   
                thickness=5
            )
            
        for pxl in self.right_arm_pxls:
            cv2.circle(
                self.frame,
                center=pxl.astype(int),
                radius=5,
                color=(255, 0, 0),   
                thickness=5
            )
            
        cv2.imshow('frame', self.frame)
        cv2.waitKey(1)
    
    @property
    def left_arm(self):
        return self.arm(self.left_arm_pxls)
    
    @property
    def right_arm(self):
        return self.arm(self.right_arm_pxls)
    