import numpy as np

class PoseClassifier:
    def __init__(self):
        # Define some example poses for classification
        self.poses = {
            "T-pose": self.is_t_pose,
            "Hands Up": self.is_hands_up,
            "Hands Down": self.is_hands_down,
            "Standing": self.is_standing,
            "Sitting": self.is_sitting
        }

    def classify(self, keypoints: np.ndarray) -> str:
        for pose_name, pose_func in self.poses.items():
            if pose_func(keypoints):
                return pose_name
        return "Unknown"

    def is_t_pose(self, keypoints: np.ndarray) -> bool:
        if len(keypoints) < 11:
            return False
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        return (abs(left_wrist[1] - left_shoulder[1]) < 20 and
                abs(right_wrist[1] - right_shoulder[1]) < 20 and
                abs(left_wrist[0] - right_wrist[0]) > 100)

    def is_hands_up(self, keypoints: np.ndarray) -> bool:
        if len(keypoints) < 11:
            return False
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        nose = keypoints[0]
        return left_wrist[1] < nose[1] and right_wrist[1] < nose[1]

    def is_hands_down(self, keypoints: np.ndarray) -> bool:
        if len(keypoints) < 13:
            return False
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        return left_wrist[1] > left_hip[1] and right_wrist[1] > right_hip[1]

    def is_standing(self, keypoints: np.ndarray) -> bool:
        if len(keypoints) < 15:
            return False
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        return (left_hip[1] < left_knee[1] and right_hip[1] < right_knee[1])

    def is_sitting(self, keypoints: np.ndarray) -> bool:
        if len(keypoints) < 15:
            return False
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        return (left_hip[1] > left_knee[1] and right_hip[1] > right_knee[1])