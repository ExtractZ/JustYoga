from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time

app = Flask(__name__)

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

POSE_WEIGHTS = {
    11: 1.2, 12: 1.2, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0,
    23: 1.5, 24: 1.5, 25: 0.8, 26: 0.8, 27: 0.7, 28: 0.7,
    29: 0.5, 30: 0.5, 31: 0.5, 32: 0.5
}


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


pose_library = PoseLibrary('poses')
pose_library.cycle_pose()


def calculate_match_percentage(detected_landmarks, reference_landmarks):
    """Calculate match percentage based on weighted Euclidean distance."""
    if not detected_landmarks or not reference_landmarks:
        return 0

    total_weight = 0
    weighted_distance_sum = 0

    for idx, (rx, ry) in reference_landmarks.items():
        if idx in detected_landmarks:
            dx, dy = detected_landmarks[idx]
            distance = np.sqrt((rx - dx) ** 2 + (ry - dy) ** 2)
            weight = POSE_WEIGHTS.get(idx, 1.0)  # Default weight is 1.0
            weighted_distance_sum += weight * distance
            total_weight += weight

    # Normalize the score to a percentage
    match_score = max(0, 100 - (weighted_distance_sum / total_weight * 100))
    return round(match_score, 2)


def generate_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        reference_landmarks = pose_library.get_reference_landmarks(pose_library.current_pose_name)

        # Draw Mediapipe pose landmarks
        detected_landmarks = {}
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

            for idx, lm in enumerate(results.pose_landmarks.landmark):
                detected_landmarks[idx] = (lm.x, lm.y)

        # Draw reference landmarks with connections
        points = {}
        if reference_landmarks:
            for idx, (x, y) in reference_landmarks.items():
                px = int(x * frame.shape[1])
                py = int(y * frame.shape[0])
                points[idx] = (px, py)
                cv2.circle(frame, (px, py), 8, (0, 255, 255), -1)

            # Draw connections
            connections = [
                (11, 13), (13, 15),  # Left arm
                (12, 14), (14, 16),  # Right arm
                (11, 12),  # Shoulders
                (23, 24),  # Hips
                (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
                (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
            ]
            for connection in connections:
                if connection[0] in points and connection[1] in points:
                    cv2.line(frame, points[connection[0]], points[connection[1]], (0, 255, 255), 2)

        # Calculate match percentage
        match_percentage = calculate_match_percentage(detected_landmarks, reference_landmarks)

        # Display match percentage
        cv2.putText(frame, f"Match: {match_percentage}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Encode the frame for streaming
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")




@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/cycle_pose", methods=["POST"])
def cycle_pose():
    next_pose = pose_library.cycle_pose()
    return jsonify({"pose_name": next_pose})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
