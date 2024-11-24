from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
from typing import Dict, Tuple
from scipy.interpolate import splprep, splev


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
pose_tracker = PoseTracker(threshold=80.0, hold_time=5.0)
pose_library.cycle_pose()



# ... (keep existing imports and setup code)

def create_body_contour(points: Dict[int, Tuple[int, int]], height: int, width: int):
    """Create a cartoonish body contour with a neck connecting the head and body"""

    def get_body_proportions() -> dict:
        """Get cartoonish body proportions"""
        height_unit = height * 0.01
        return {
            'head_radius': height_unit * 8.0,
            'body_width': height_unit * 18.0,
            'limb_thickness': height_unit * 7.0,
            'neck_width': height_unit * 6.0,  # Width of the neck
            'neck_height': height_unit * 7.0  # Height of the neck
        }

    def create_rounded_rectangle(top_center, bottom_center, top_width, bottom_width, roundness=0.1):
        """
        Create a rounded rectangle shape using interpolated points and smooth corners
        with Bezier curve approximation for better rounding.
        """
        # Calculate corner points
        top_left = (top_center[0] - top_width // 2, top_center[1])
        top_right = (top_center[0] + top_width // 2, top_center[1])
        bottom_left = (bottom_center[0] - bottom_width // 2, bottom_center[1])
        bottom_right = (bottom_center[0] + bottom_width // 2, bottom_center[1])

        height = bottom_center[1] - top_center[1]
        corner_radius = int(min(top_width, bottom_width, height) * roundness)
        
        # Number of points to generate for each curved section
        curve_points = 15  # Increased number of points for smoother curves
        
        points = []
        
        def create_bezier_curve(p0, p1, p2, num_points):
            """Create a quadratic Bezier curve between points"""
            curve = []
            for t in np.linspace(0, 1, num_points):
                # Quadratic Bezier formula
                x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
                y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
                curve.append([int(x), int(y)])
            return curve

        def create_edge_points(start, end, num_points):
            """Create smoothly interpolated points along an edge"""
            points = []
            for t in np.linspace(0, 1, num_points):
                x = int(start[0] + (end[0] - start[0]) * t)
                y = int(start[1] + (end[1] - start[1]) * t)
                points.append([x, y])
            return points

        # Top edge with rounded corners
        top_left_corner = (top_left[0] + corner_radius, top_left[1])
        top_right_corner = (top_right[0] - corner_radius, top_right[1])
        
        # Top-left corner curve
        control_point_tl = (top_left[0], top_left[1])
        points.extend(create_bezier_curve(
            (top_left[0], top_left[1] + corner_radius),
            control_point_tl,
            top_left_corner,
            curve_points
        ))
        
        # Top edge
        points.extend(create_edge_points(top_left_corner, top_right_corner, 20))
        
        # Top-right corner curve
        control_point_tr = (top_right[0], top_right[1])
        points.extend(create_bezier_curve(
            top_right_corner,
            control_point_tr,
            (top_right[0], top_right[1] + corner_radius),
            curve_points
        ))
        
        # Right edge
        right_edge_start = (top_right[0], top_right[1] + corner_radius)
        right_edge_end = (bottom_right[0], bottom_right[1] - corner_radius)
        points.extend(create_edge_points(right_edge_start, right_edge_end, 30))
        
        # Bottom-right corner curve
        control_point_br = (bottom_right[0], bottom_right[1])
        points.extend(create_bezier_curve(
            (bottom_right[0], bottom_right[1] - corner_radius),
            control_point_br,
            (bottom_right[0] - corner_radius, bottom_right[1]),
            curve_points
        ))
        
        # Bottom edge
        points.extend(create_edge_points(
            (bottom_right[0] - corner_radius, bottom_right[1]),
            (bottom_left[0] + corner_radius, bottom_left[1]),
            20
        ))
        
        # Bottom-left corner curve
        control_point_bl = (bottom_left[0], bottom_left[1])
        points.extend(create_bezier_curve(
            (bottom_left[0] + corner_radius, bottom_left[1]),
            control_point_bl,
            (bottom_left[0], bottom_left[1] - corner_radius),
            curve_points
        ))
        
        # Left edge
        left_edge_start = (bottom_left[0], bottom_left[1] - corner_radius)
        left_edge_end = (top_left[0], top_left[1] + corner_radius)
        points.extend(create_edge_points(left_edge_start, left_edge_end, 30))

        return np.array(points)

    # Initialize mask
    contour_mask = np.zeros((height, width), dtype=np.uint8)

    aura_mask = np.zeros((height, width), dtype=np.uint8)
    inner_aura_mask = np.zeros((height, width), dtype=np.uint8)


    props = get_body_proportions()

    # Calculate head position dynamically
    torso_landmarks = [11, 12, 23, 24]  # shoulders and hips
    if all(idx in points for idx in torso_landmarks):
        # Calculate shoulder and hip centers
        shoulders_center = np.array((np.array(points[11]) + np.array(points[12])) // 2)
        hips_center = np.array((np.array(points[23]) + np.array(points[24])) // 2)

    
        hips_center[1] += int(props['neck_height'] * 0.5)
        shoulders_center[1] -= int(props['neck_height'] * 0.5)

        # Interpolate the head position along the line formed by shoulders and hips
        torso_vector = shoulders_center - hips_center
        torso_length = np.linalg.norm(torso_vector)
        torso_unit_vector = torso_vector / torso_length if torso_length > 0 else np.array([0, -1])

        head_position = shoulders_center + torso_unit_vector * props['head_radius'] * 1
        head_position = tuple(map(int, head_position))

        # Draw head
        cv2.circle(contour_mask, head_position, int(props['head_radius']), 255, -1, cv2.LINE_AA)
        cv2.circle(aura_mask, head_position, int(props['head_radius']), 255, -1, cv2.LINE_AA)

        # Calculate neck dimensions
        neck_top_center = (shoulders_center[0], shoulders_center[1] - int(props['neck_height'] / 2))
        neck_bottom_center = (shoulders_center[0], shoulders_center[1] + int(props['neck_height'] / 2))

        neck_points = create_rounded_rectangle(
            tuple(map(int, neck_top_center)),
            tuple(map(int, neck_bottom_center)),
            int(props['neck_width']),
            int(props['neck_width']),
            roundness=0.3
        )

        # Draw neck
        if len(neck_points) > 0:
            cv2.fillPoly(contour_mask, [neck_points], 255, cv2.LINE_AA)
            cv2.fillPoly(aura_mask, [neck_points], 255, cv2.LINE_AA)

        # Create rounded body shape
        body_points = create_rounded_rectangle(
            tuple(map(int, shoulders_center)),
            tuple(map(int, hips_center)),
            int(props['body_width']),
            int(props['body_width'] * 0.8),
            roundness=0.3
        )

        # Draw body
        if len(body_points) > 0:
            cv2.fillPoly(contour_mask, [body_points], 255, cv2.LINE_AA)
            cv2.fillPoly(aura_mask, [body_points], 255, cv2.LINE_AA)

    # Draw limbs
    def draw_limb(start_idx, mid_idx, end_idx):
        if all(idx in points for idx in [start_idx, mid_idx, end_idx]):
            thickness = int(props['limb_thickness'])

            # Draw segments
            cv2.line(contour_mask, points[start_idx], points[mid_idx], 255, thickness, cv2.LINE_AA)
            cv2.line(contour_mask, points[mid_idx], points[end_idx], 255, thickness, cv2.LINE_AA)

            # Add joint circles for smooth connections
            cv2.circle(contour_mask, points[mid_idx], thickness // 2, 255, -1, cv2.LINE_AA)
            
            cv2.line(aura_mask, points[start_idx], points[mid_idx], 255, thickness, cv2.LINE_AA)
            cv2.line(aura_mask, points[mid_idx], points[end_idx], 255, thickness, cv2.LINE_AA)
            cv2.circle(aura_mask, points[mid_idx], thickness // 2, 255, -1, cv2.LINE_AA)


    # Draw limbs
    draw_limb(11, 13, 15)  # Left arm
    draw_limb(12, 14, 16)  # Right arm
    draw_limb(23, 25, 27)  # Left leg
    draw_limb(24, 26, 28)  # Right leg

    
    # Threshold the aura to make it more pronounced
    inner_aura_blur = cv2.GaussianBlur(aura_mask, (21, 21), 5)
    _, inner_aura_thresh = cv2.threshold(inner_aura_blur, 30, 255, cv2.THRESH_BINARY)


    outer_aura_blur = cv2.GaussianBlur(aura_mask, (45, 45), 15)
    _, outer_aura_thresh = cv2.threshold(outer_aura_blur, 15, 255, cv2.THRESH_BINARY)
    
    # Final smoothing
    kernel_size = 5
    contour_mask = cv2.GaussianBlur(contour_mask, (kernel_size, kernel_size), 2)
    _, contour_mask = cv2.threshold(contour_mask, 127, 255, cv2.THRESH_BINARY)

    return contour_mask, inner_aura_thresh, outer_aura_thresh

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

        height, width = frame.shape[:2]
        
        # Create points dictionary for reference pose
        reference_points = {}
        for idx, (x, y) in reference_landmarks.items():
            px = int(x * width)
            py = int(y * height)
            reference_points[idx] = (px, py)
        
        # Create contour and aura masks
        contour_mask, inner_aura_mask, outer_aura_mask = create_body_contour(reference_points, height, width)
        
        # Create colored overlays
        contour_overlay = np.zeros_like(frame, dtype=np.uint8)
        inner_aura_overlay = np.zeros_like(frame, dtype=np.uint8)
        outer_aura_overlay = np.zeros_like(frame, dtype=np.uint8)
        
        # Calculate overall match before setting colors
        overall_match = 0
        if results.pose_landmarks:
            landmark_overlay = np.zeros_like(frame, dtype=np.uint8)
            
            mp_drawing.draw_landmarks(
                landmark_overlay, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=1)
            )
            
            frame = cv2.addWeighted(frame, 1, landmark_overlay, 0.0, 0)
            
            distances = calculate_landmark_distances(results.pose_landmarks, reference_landmarks)
            overall_match = calculate_pose_similarity(distances)
            pose_tracker.update(overall_match)

        # Set colors based on overall_match
        if overall_match >= 80.0:
            # Colors when pose matches or exceeds threshold
            contour_overlay[contour_mask > 0] = [0, 255, 0]  # Bright green
            inner_aura_overlay[inner_aura_mask > 0] = [50, 255, 50]  # Golden yellow
            outer_aura_overlay[outer_aura_mask > 0] = [50, 255, 50]  # Orange
        else:
            # Default colors when pose doesn't match threshold
            contour_overlay[contour_mask > 0] = [0, 0, 255]  # Bright red
            inner_aura_overlay[inner_aura_mask > 0] = [50, 50, 255]  # White-green
            outer_aura_overlay[outer_aura_mask > 0] = [50, 50, 255]  # Light green
        
        # Layer the aura effects with higher opacity
        # Add outer aura first (more transparent)
        frame = cv2.addWeighted(frame, 1, outer_aura_overlay, 0.4, 0)
        
        # Add inner aura (more opaque)
        frame = cv2.addWeighted(frame, 1, inner_aura_overlay, 0.6, 0)
        
        # Add main contour last
        frame = cv2.addWeighted(frame, 1, contour_overlay, 0.5, 0)

        draw_status_overlay(frame, overall_match, pose_tracker, pose_library.current_pose_name)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()

# ... (keep the rest of the code the same)

# Keep the original Flask routes unchanged
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

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