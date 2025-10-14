"""pose_arm_coords.py

Read a human pose (JSON / npz / simple text) and output 3D coordinates of arm joints.

Supported input formats:
- MediaPipe landmarks JSON: {"people": [...]} or a single dict with "landmark" list of 33 landmarks
- Simple JSON mapping joint names to [x,y,z]
- NPZ containing an array named `landmarks` of shape (N,3)

Outputs coordinates for joints:
- left_shoulder, left_elbow, left_wrist
- right_shoulder, right_elbow, right_wrist

Usage:
    python -m utils.pose_arm_coords /path/to/pose.json

If run as a script, prints JSON with the joint coordinates.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

MEDIA_PIPE_NAMES = [
    'nose','left_eye_inner','left_eye','left_eye_outer','right_eye_inner','right_eye','right_eye_outer',
    'left_ear','right_ear','mouth_left','mouth_right','left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_pinky','right_pinky','left_index','right_index','left_thumb','right_thumb',
    'right_hip','left_hip','right_knee','left_knee','right_ankle','left_ankle','right_heel','left_heel',
    'right_foot_index','left_foot_index'
]

ARM_JOINT_KEYS = {
    'left': ('left_shoulder', 'left_elbow', 'left_wrist'),
    'right': ('right_shoulder', 'right_elbow', 'right_wrist'),
}


def load_pose(path: Path) -> Dict[str, Any]:
    s = path.suffix.lower()
    if s == '.json':
        return json.loads(path.read_text())
    elif s in ('.npz',):
        data = np.load(str(path))
        return {k: data[k] for k in data.files}
    else:
        # try plain text -> JSON
        try:
            return json.loads(path.read_text())
        except Exception:
            raise ValueError(f"Unsupported file type: {s}")


def extract_landmarks_from_mediapipe(obj: Any) -> Optional[np.ndarray]:
    # Case 1: MediaPipe holistic: {'landmark': [{'x':..,'y':..,'z':..}, ...]}
    if isinstance(obj, dict) and 'landmark' in obj:
        lm = obj['landmark']
        pts = []
        for l in lm:
            x = l.get('x', 0)
            y = l.get('y', 0)
            z = l.get('z', 0)
            pts.append([x, y, z])
        return np.array(pts)

    # Case 2: Top-level dict with 'people' list (openpose style) or mediapipe results saved
    if isinstance(obj, dict) and 'people' in obj:
        people = obj['people']
        if not people:
            return None
        p = people[0]
        if isinstance(p, dict) and 'pose_keypoints_2d' in p:
            arr = np.array(p['pose_keypoints_2d']).reshape(-1, 3)[:, :2]
            # expand z=0
            z = np.zeros((arr.shape[0], 1))
            return np.hstack([arr, z])

    # Case 3: direct mapping from names to coordinates
    if isinstance(obj, dict):
        # find known names
        coords = {}
        for k, v in obj.items():
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                coords[k] = np.array(v[:3])
        if coords:
            # return array aligned to MEDIA_PIPE_NAMES where possible
            pts = np.zeros((len(MEDIA_PIPE_NAMES), 3), dtype=float)
            found = False
            for idx, name in enumerate(MEDIA_PIPE_NAMES):
                if name in coords:
                    pts[idx] = coords[name]
                    found = True
            if found:
                return pts

    # Case 4: NPZ-like structure passed as dict with 'landmarks' key
    if isinstance(obj, dict) and 'landmarks' in obj:
        lm = obj['landmarks']
        return np.array(lm)

    return None


def get_arm_joint_coords(landmarks: np.ndarray) -> Dict[str, np.ndarray]:
    """landmarks: (N,3) array, expected to follow MEDIA_PIPE_NAMES ordering for known indices.
    Returns dictionary mapping joint names to 3D coordinates.
    """
    out = {}
    name_to_idx = {name: i for i, name in enumerate(MEDIA_PIPE_NAMES)}
    for side in ('left', 'right'):
        s_keys = ARM_JOINT_KEYS[side]
        coords = []
        for key in s_keys:
            idx = name_to_idx.get(key, None)
            if idx is None or idx >= len(landmarks):
                coords.append([None, None, None])
            else:
                coords.append(landmarks[idx].tolist())
        out[side] = dict(zip(s_keys, coords))
    return out


def main(argv):
    if len(argv) < 2:
        print('Usage: python -m utils.pose_arm_coords /path/to/pose.json')
        return 1
    path = Path(argv[1])
    if not path.exists():
        print('File not found:', path)
        return 1

    obj = load_pose(path)
    landmarks = None

    # If file was an npz with array named landmarks
    if isinstance(obj, dict) and any(k in ('landmarks','points','pose') for k in obj.keys()):
        for k in ('landmarks','points','pose'):
            if k in obj:
                landmarks = np.array(obj[k])
                break

    if landmarks is None:
        # try mediapipe extract
        landmarks = extract_landmarks_from_mediapipe(obj)

    if landmarks is None:
        raise ValueError('Could not extract landmarks from input')

    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(-1, 3)

    arm_coords = get_arm_joint_coords(landmarks)

    # Print JSON-friendly output
    printable = {}
    for side, d in arm_coords.items():
        printable[side] = {k: (v if v is None or any(x is None for x in v) else [float(x) for x in v]) for k, v in d.items()}

    print(json.dumps(printable, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
