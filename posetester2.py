import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple
import json
import os
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Define importance weights for different body parts globally
POSE_WEIGHTS = {
    11: 1.2,  # shoulders
    12: 1.2,
    13: 1.0,  # elbows
    14: 1.0,
    15: 1.0,  # wrists
    16: 1.0,
    23: 1.5,  # hips
    24: 1.5,
    25: 0.8,  # knees
    26: 0.8,
    27: 0.7,  # ankles
    28: 0.7,
    29: 0.5,  # heels
    30: 0.5,
    31: 0.5,  # foot index
    32: 0.5,
}

class PoseTracker:
    def __init__(self, threshold=30.0, hold_time=5.0):
        self.threshold = threshold
        self.required_hold_time = hold_time
        self.start_time = None
        self.is_holding = False
        self.completed = False
        self.elapsed_time = 0.0

    def update(self, similarity: float) -> bool:
        """Update the pose tracking state"""
        current_time = time.time()
        
        if similarity >= self.threshold:
            if not self.is_holding:
                # Just started holding the pose
                self.start_time = current_time
                self.is_holding = True
            else:
                # Continue holding the pose
                self.elapsed_time = current_time - self.start_time
                if self.elapsed_time >= self.required_hold_time:
                    self.completed = True
        else:
            # Reset if pose is broken
            self.is_holding = False
            self.start_time = None
            self.elapsed_time = 0.0
            
        return self.completed

    def reset(self):
        """Reset the tracker state"""
        self.start_time = None
        self.is_holding = False
        self.completed = False
        self.elapsed_time = 0.0

class PoseLibrary:
    def __init__(self, poses_directory: str):
        """Initialize the pose library from a directory of pose files"""
        self.poses = {}
        self.current_pose_name = None
        self.load_poses(poses_directory)
    
    def load_poses(self, directory: str):
        """Load all pose files from the specified directory"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created poses directory: {directory}")
            return

        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                pose_name = filename[:-5].upper()  # Remove .json and convert to uppercase
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as f:
                    # Convert string keys to integers when loading
                    pose_data = json.load(f)
                    self.poses[pose_name] = {int(k): v for k, v in pose_data.items()}
    
    def get_reference_landmarks(self, pose_name: str = None) -> Dict[int, Tuple[float, float]]:
        """Get reference landmarks for the specified pose"""
        if pose_name is None:
            pose_name = self.current_pose_name
        if pose_name not in self.poses:
            raise ValueError(f"Pose '{pose_name}' not found in library")
        return self.poses[pose_name]
    
    def cycle_pose(self):
        """Cycle to the next available pose"""
        pose_names = list(self.poses.keys())
        if not self.current_pose_name:
            self.current_pose_name = pose_names[0] if pose_names else None
        else:
            current_index = pose_names.index(self.current_pose_name)
            next_index = (current_index + 1) % len(pose_names)
            self.current_pose_name = pose_names[next_index]
        return self.current_pose_name

def save_pose_to_file(landmarks, filename: str):
    """Save the current pose landmarks to a JSON file"""
    landmark_dict = {}
    for idx in POSE_WEIGHTS.keys():
        if idx >= len(landmarks.landmark):
            continue
        landmark = landmarks.landmark[idx]
        landmark_dict[idx] = (landmark.x, landmark.y)
    
    with open(filename, 'w') as f:
        json.dump(landmark_dict, f, indent=2)

def calculate_landmark_distances(current_landmarks, reference_landmarks: Dict[int, Tuple[float, float]]) -> Dict[int, float]:
    """Calculate distances between current and reference landmarks with weights"""
    distances = {}
    
    for idx, ref_pos in reference_landmarks.items():
        if idx >= len(current_landmarks.landmark):
            continue
            
        current = current_landmarks.landmark[idx]
        distance = np.sqrt(
            (ref_pos[0] - current.x) ** 2 +
            (ref_pos[1] - current.y) ** 2
        )
        weighted_distance = distance * POSE_WEIGHTS.get(idx, 1.0)
        distances[idx] = weighted_distance
        
    return distances

def calculate_pose_similarity(distances: Dict[int, float]) -> float:
    """Calculate overall pose similarity using weighted average"""
    if not distances:
        return 0.0
    
    total_weight = sum(POSE_WEIGHTS.get(idx, 1.0) for idx in distances.keys())
    weighted_avg_distance = sum(dist * POSE_WEIGHTS.get(idx, 1.0) 
                              for idx, dist in distances.items()) / total_weight
    
    similarity = 100 * np.exp(-5 * weighted_avg_distance)
    similarity = np.power(similarity / 100, 1.2) * 100
    
    return max(0, min(100, similarity))

def draw_reference_overlay_with_distances(frame: np.ndarray, 
                                       current_landmarks,
                                       reference_landmarks: Dict[int, Tuple[float, float]],
                                       pose_type: str,
                                       pose_tracker: PoseTracker,
                                       alpha: float = 0.3) -> np.ndarray:
    """Draw reference pose and distance measurements"""
    height, width = frame.shape[:2]
    overlay = np.zeros_like(frame, dtype=np.uint8)
    
    distances = {}
    overall_match = 0
    if current_landmarks:
        distances = calculate_landmark_distances(current_landmarks, reference_landmarks)
        overall_match = calculate_pose_similarity(distances)
    
    # Update pose tracker
    pose_completed = pose_tracker.update(overall_match)
    
    points = {}
    for idx, (x, y) in reference_landmarks.items():
        px = int(x * width)
        py = int(y * height)
        points[idx] = (px, py)
        
        if idx in distances:
            match_quality = np.exp(-5 * distances[idx])
            color = (
                int(255 * (1 - match_quality)),
                int(255 * match_quality),
                0
            )
            cv2.circle(overlay, (px, py), 8, color, -1)
            
            if distances[idx] > 0.1:
                dist_text = f"{distances[idx]:.2f}"
                cv2.putText(overlay, dist_text, (px + 10, py),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.circle(overlay, (px, py), 8, (0, 255, 255), -1)
    
    connections = [
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 12), (11, 23), (12, 24), (23, 24),
        (23, 25), (24, 26),
        (25, 27), (26, 28),
        (27, 29), (28, 30),
        (29, 31), (30, 32),
    ]
    
    for connection in connections:
        if connection[0] in points and connection[1] in points:
            start = points[connection[0]]
            end = points[connection[1]]
            cv2.line(overlay, start, end, (255, 255, 255), 3)
    
    labels = [
        f"Reference Pose: {pose_type}",
        f"Overall Match: {overall_match:.1f}%"
    ]
    
    if pose_tracker.is_holding:
        remaining_time = max(0, pose_tracker.required_hold_time - pose_tracker.elapsed_time)
        labels.append(f"Hold Time: {remaining_time:.1f}s")
        
        # Draw progress bar
        bar_width = 200
        bar_height = 20
        progress = min(1.0, pose_tracker.elapsed_time / pose_tracker.required_hold_time)
        filled_width = int(bar_width * progress)
        
        # Draw background bar
        cv2.rectangle(overlay, 
                     (10, height - 40), 
                     (10 + bar_width, height - 40 + bar_height),
                     (100, 100, 100), -1)
        
        # Draw filled portion
        cv2.rectangle(overlay, 
                     (10, height - 40),
                     (10 + filled_width, height - 40 + bar_height),
                     (0, 255, 0), -1)
    
    if pose_completed:
        labels.append("Great job! Pose completed!")
    
    y_offset = 30
    for label in labels:
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(overlay, (10, y_offset - text_height - 5), 
                     (text_width + 20, y_offset + 5), (0, 0, 0), -1)
        cv2.putText(overlay, label, (15, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_offset += 40
    
    result = cv2.addWeighted(frame, 1, overlay, alpha, 0)
    return result

def getCamera():
    for n in range(6):
        cap = cv2.VideoCapture(n)
        if cap.isOpened():
            print(f"using camera index {n}")
            return cap
        cap.release()
    raise ValueError("none of the cameras work")

def main():
    # Initialize pose library
    pose_library = PoseLibrary('poses')
    recording_mode = False
    
    # Initialize pose tracker
    pose_tracker = PoseTracker(threshold=30.0, hold_time=5.0)
    
    try:
        cap = getCamera()
        cv2.namedWindow("Yoga Pose Practice", cv2.WINDOW_NORMAL)
    except ValueError as e:
        print(e)
        return

    current_pose = pose_library.cycle_pose()

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
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            )
        
        reference_landmarks = pose_library.get_reference_landmarks(current_pose)
        
        frame_with_overlay = draw_reference_overlay_with_distances(
            frame, 
            results.pose_landmarks if results.pose_landmarks else None,
            reference_landmarks,
            current_pose,
            pose_tracker,
            alpha=0.3
        )
        
        if recording_mode:
            cv2.putText(frame_with_overlay, "Recording Mode - Press 's' to save pose",
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 2)
        
        cv2.imshow("Yoga Pose Practice", frame_with_overlay)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_pose = pose_library.cycle_pose()
            pose_tracker.reset()  # Reset tracker when changing poses
        elif key == ord('r'):
            recording_mode = not recording_mode
        elif key == ord('s') and recording_mode and results.pose_landmarks:
            pose_name = input("Enter pose name: ")
            if pose_name:
                filename = os.path.join('poses', f"{pose_name.lower()}.json")
                save_pose_to_file(results.pose_landmarks, filename)
                print(f"Saved pose to {filename}")
                pose_library.load_poses('poses')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()