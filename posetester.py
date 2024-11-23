import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Define importance weights for different body parts globally
POSE_WEIGHTS = {
    # Upper body (more important for most poses)
    11: 1.2,  # shoulders
    12: 1.2,
    13: 1.0,  # elbows
    14: 1.0,
    15: 1.0,  # wrists
    16: 1.0,
    # Core (very important for stability)
    23: 1.5,  # hips
    24: 1.5,
    # Lower body
    25: 0.8,  # knees
    26: 0.8,
    27: 0.7,  # ankles
    28: 0.7,
    29: 0.5,  # heels
    30: 0.5,
    31: 0.5,  # foot index
    32: 0.5,
}

def get_reference_landmarks() -> Dict[int, Tuple[float, float]]:
    """Return the reference pose landmark positions"""
    return {
        # Upper body landmarks
        11: (0.45, 0.2),  # left shoulder
        12: (0.55, 0.2),  # right shoulder
        13: (0.35, 0.35), # left elbow
        14: (0.65, 0.35), # right elbow
        15: (0.35, 0.5),  # left wrist
        16: (0.65, 0.5),  # right wrist
        
        # Lower body landmarks
        23: (0.45, 0.6),  # left hip
        24: (0.55, 0.6),  # right hip
        25: (0.45, 0.75), # left knee
        26: (0.55, 0.75), # right knee
        27: (0.45, 0.9),  # left ankle
        28: (0.55, 0.9),  # right ankle
        29: (0.45, 0.95), # left heel
        30: (0.55, 0.95), # right heel
        31: (0.45, 0.98), # left foot index
        32: (0.55, 0.98), # right foot index
    }

def calculate_landmark_distances(current_landmarks, reference_landmarks: Dict[int, Tuple[float, float]]) -> Dict[int, float]:
    """Calculate distances between current and reference landmarks with weights"""
    distances = {}
    
    for idx, ref_pos in reference_landmarks.items():
        if idx >= len(current_landmarks.landmark):
            continue
            
        current = current_landmarks.landmark[idx]
        
        # Calculate Euclidean distance between normalized coordinates
        distance = np.sqrt(
            (ref_pos[0] - current.x) ** 2 +
            (ref_pos[1] - current.y) ** 2
        )
        
        # Apply weight to the distance
        weighted_distance = distance * POSE_WEIGHTS.get(idx, 1.0)
        distances[idx] = weighted_distance
        
    return distances

def calculate_pose_similarity(distances: Dict[int, float]) -> float:
    """Calculate overall pose similarity using a more sophisticated method"""
    if not distances:
        return 0.0
    
    # Calculate the weighted average distance
    total_weight = sum(POSE_WEIGHTS.get(idx, 1.0) for idx in distances.keys())
    weighted_avg_distance = sum(dist * POSE_WEIGHTS.get(idx, 1.0) 
                              for idx, dist in distances.items()) / total_weight
    
    # Convert distance to similarity score using exponential decay
    # This gives a more intuitive curve than linear scaling
    similarity = 100 * np.exp(-5 * weighted_avg_distance)
    
    # Adjust the curve to be more stringent
    # This will make it harder to get very high scores
    similarity = np.power(similarity / 100, 1.2) * 100
    
    return max(0, min(100, similarity))

def draw_reference_overlay_with_distances(frame: np.ndarray, 
                                       current_landmarks, 
                                       pose_type: str, 
                                       alpha: float = 0.3) -> np.ndarray:
    """Draw reference pose and distance measurements"""
    height, width = frame.shape[:2]
    overlay = np.zeros_like(frame, dtype=np.uint8)
    
    # Get reference landmarks
    reference_landmarks = get_reference_landmarks()
    
    # Calculate distances if we have current landmarks
    distances = {}
    overall_match = 0
    if current_landmarks:
        distances = calculate_landmark_distances(current_landmarks, reference_landmarks)
        overall_match = calculate_pose_similarity(distances)
    
    # Draw reference pose
    points = {}
    for idx, (x, y) in reference_landmarks.items():
        px = int(x * width)
        py = int(y * height)
        points[idx] = (px, py)
        
        # Color the landmark based on distance if available
        if idx in distances:
            # Use exponential scaling for color gradient
            match_quality = np.exp(-5 * distances[idx])
            color = (
                int(255 * (1 - match_quality)),  # B
                int(255 * match_quality),        # G
                0                                # R
            )
            cv2.circle(overlay, (px, py), 8, color, -1)
            
            # Draw distance value near the landmark
            if distances[idx] > 0.1:  # Only show significant deviations
                dist_text = f"{distances[idx]:.2f}"
                cv2.putText(overlay, dist_text, (px + 10, py),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.circle(overlay, (px, py), 8, (0, 255, 255), -1)
    
    # Draw connections
    connections = [
        # Arms
        (11, 13), (13, 15), # Left arm
        (12, 14), (14, 16), # Right arm
        # Torso
        (11, 12), (11, 23), (12, 24), (23, 24),
        # Legs
        (23, 25), (24, 26),
        (25, 27), (26, 28),
        (27, 29), (28, 30),
        (29, 31), (30, 32),
    ]
    
    for connection in connections:
        start = points[connection[0]]
        end = points[connection[1]]
        cv2.line(overlay, start, end, (255, 255, 255), 3)
    
    # Add text labels with dark background
    labels = [
        f"Reference Pose: {pose_type}",
        f"Overall Match: {overall_match:.1f}%"
    ]
    
    y_offset = 30
    for label in labels:
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(overlay, (10, y_offset - text_height - 5), 
                     (text_width + 20, y_offset + 5), (0, 0, 0), -1)
        cv2.putText(overlay, label, (15, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_offset += 40
    
    # Blend the overlay with the original frame
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

try:
    cap = getCamera()
    cv2.namedWindow("Yoga Pose Practice", cv2.WINDOW_NORMAL)
except ValueError as e:
    print(e)
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        # Draw the pose landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            )
        
        # Add the reference pose overlay with distances
        frame_with_overlay = draw_reference_overlay_with_distances(
            frame, 
            results.pose_landmarks if results.pose_landmarks else None,
            "MOUNTAIN",
            alpha=0.3
        )
        
        # Display the result
        cv2.imshow("Yoga Pose Practice", frame_with_overlay)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()