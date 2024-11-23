from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
from typing import Dict, Tuple

app = Flask(__name__)

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Enhanced pose weights with better anatomical consideration
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

# Enhanced connections for better pose visualization
POSE_CONNECTIONS = [
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 12), (11, 23), (12, 24), (23, 24),  # Torso
    (23, 25), (24, 26),  # Upper legs
    (25, 27), (26, 28),  # Lower legs
    (27, 29), (28, 30),  # Ankles to heels
    (29, 31), (30, 32),  # Heels to toes
]

class PoseTracker:
    def __init__(self, threshold=80.0, hold_time=5.0):
        self.threshold = threshold
        self.required_hold_time = hold_time
        self.start_time = None
        self.is_holding = False
        self.completed = False
        self.elapsed_time = 0.0
        self.best_match = 0.0


    def update(self, similarity: float) -> bool:
        """Update pose tracking state with current similarity score"""
        current_time = time.time()
        self.best_match = max(self.best_match, similarity)
        
        if similarity >= self.threshold:
            if not self.is_holding:
                self.start_time = current_time
                self.is_holding = True

            else:
                self.elapsed_time = current_time - self.start_time
                if self.elapsed_time >= self.required_hold_time and not self.completed:
                    self.completed = True

        else:
            self.is_holding = False
            self.start_time = None
            self.elapsed_time = 0.0
            
        return self.completed

    def get_progress(self) -> float:
        """Get current progress as a percentage"""
        if not self.is_holding:
            return 0.0
        return min(100.0, (self.elapsed_time / self.required_hold_time) * 100)

    def get_stats(self) -> Dict:
        """Get current tracking statistics"""
        return {
            "best_match": self.best_match,

        }

    def reset(self):
        """Reset tracker state"""
        self.start_time = None
        self.is_holding = False
        self.completed = False
        self.elapsed_time = 0.0
        self.best_match = 0.0

class PoseLibrary:
    def __init__(self, poses_directory):
        self.poses = {}
        self.current_pose_name = None
        self.load_poses(poses_directory)

    def load_poses(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                pose_name = filename[:-5].upper()
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as f:
                    pose_data = json.load(f)
                    self.poses[pose_name] = {int(k): v for k, v in pose_data.items()}

    def get_reference_landmarks(self, pose_name=None):
        if pose_name is None:
            pose_name = self.current_pose_name
        if pose_name not in self.poses:
            raise ValueError(f"Pose '{pose_name}' not found in library")
        return self.poses[pose_name]

    def cycle_pose(self):
        pose_names = list(self.poses.keys())
        if not self.current_pose_name:
            self.current_pose_name = pose_names[0] if pose_names else None
        else:
            current_index = pose_names.index(self.current_pose_name)
            next_index = (current_index + 1) % len(pose_names)
            self.current_pose_name = pose_names[next_index]
        return self.current_pose_name

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

def draw_loading_pie(frame, progress, center, radius, color=(0, 255, 0), thickness=-1):
    """Draw a pie chart representing the loading progress"""
    angle = (360 * (1 - progress / 100))  # Calculate angle for ellipse
    start_angle = 90  # Start at the top
    end_angle = 90 - angle  # Calculate end angle
    
    # Draw the background circle
    cv2.circle(frame, center, radius, (100, 100, 100), thickness)
    
    # Draw the filled pie
    cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, color, thickness)
    
    # Draw the border
    cv2.circle(frame, center, radius, (255, 255, 255), 2)

def draw_status_overlay(frame, match_percentage, pose_tracker, current_pose):
    """Draw status information on the frame"""
    height, width = frame.shape[:2]
    
    # Define radius before using it
    radius = 30  # Adjust the size of the pie chart as needed
    
    # Draw holding progress pie
    if pose_tracker.is_holding:
        progress = pose_tracker.get_progress()
        center = (int(width - radius - 10), height - 30)  # Adjust position as needed
        draw_loading_pie(frame, progress, center, radius)
        
    
    # Draw match percentage
    cv2.putText(frame, f"Match: {match_percentage:.1f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Draw current pose name
    cv2.putText(frame, f"Pose: {current_pose}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Draw completion message
    if pose_tracker.completed:
        cv2.putText(frame, "Great job! Pose completed!",
                    (frame.shape[1] // 2 - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw statistics
    stats = pose_tracker.get_stats()
    y_pos = 110
    cv2.putText(frame, f"Best Match: {stats['best_match']:.1f}%",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y_pos += 30


pose_library = PoseLibrary('poses')
pose_tracker = PoseTracker(threshold=20.0, hold_time=5.0)
pose_library.cycle_pose()

def generate_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Use the updated current_pose_name to fetch reference landmarks
        reference_landmarks = pose_library.get_reference_landmarks(pose_library.current_pose_name)

        # Create overlay for reference pose
        height, width = frame.shape[:2]
        overlay = np.zeros_like(frame, dtype=np.uint8)
        
        # Process and draw current pose landmarks
        overall_match = 0
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )
            
            # Calculate distances and similarity
            distances = calculate_landmark_distances(results.pose_landmarks, reference_landmarks)
            overall_match = calculate_pose_similarity(distances)
            
            # Update pose tracker
            pose_tracker.update(overall_match)

        # Draw reference landmarks with enhanced visualization
        points = {}
        for idx, (x, y) in reference_landmarks.items():
            px = int(x * width)
            py = int(y * height)
            points[idx] = (px, py)
            cv2.circle(overlay, (px, py), 8, (0, 255, 255), -1)

        # Draw enhanced connections
        for connection in POSE_CONNECTIONS:
            if connection[0] in points and connection[1] in points:
                cv2.line(overlay, points[connection[0]], points[connection[1]], (0, 255, 255), 2)

        # Add overlay with transparency
        frame = cv2.addWeighted(frame, 1, overlay, 0.3, 0)
        
        # Draw status overlay
        draw_status_overlay(frame, overall_match, pose_tracker, pose_library.current_pose_name)

        # Encode frame for streaming
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/cycle_pose", methods=["POST"])
def cycle_pose():
    next_pose = pose_library.cycle_pose()
    pose_tracker.reset()  # Reset tracker when changing poses
    return jsonify({
        "pose_name": next_pose,
        "stats": pose_tracker.get_stats()
    })

@app.route("/set_current_pose", methods=["POST"])
def set_current_pose():
    """Set the current pose for overlay."""
    data = request.get_json()
    pose_name = data.get("pose_name")

    if pose_name not in pose_library.poses:
        return jsonify({"error": f"Pose '{pose_name}' not found"}), 400

    pose_library.current_pose_name = pose_name
    return jsonify({"message": f"Current pose set to '{pose_name}'"})

@app.route("/get_pose_list", methods=["POST"])
def get_pose_list():
    """Return a list of up to 5 pose names for the circuit."""
    pose_names = list(pose_library.poses.keys())
    if not pose_names:
        return jsonify({"error": "No poses available"}), 400
    # Select up to 5 poses for the circuit
    circuit_poses = pose_names[:5]
    return jsonify({"poses": circuit_poses})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)