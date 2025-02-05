import cv2
from ultralytics import YOLO
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        # Load the YOLO model for object detection
        self.model = YOLO(model_path, task="detect")  # Ensure the task is set to "detect" for bounding boxes

    def predict(self, frame):
        """Detect court bounding boxes in the given frame."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert image from BGR (OpenCV format) to RGB
        results = self.model.predict(img_rgb, conf=1e-34)  # Run YOLO model prediction with a very low confidence threshold

        # Check if any bounding boxes were detected
        if not results or len(results[0].boxes) == 0:
            print("No bounding boxes detected.")
            return None

        # Filter detected bounding boxes to keep only the most relevant ones
        filtered_boxes = self.filter_highest_confidence_with_replacement(results[0].boxes, frame.shape[:2])
        keypoints = self.extract_bounding_box_centers(filtered_boxes)
        return keypoints

    def filter_highest_confidence_with_replacement(self, boxes, frame_shape):
        """
        Filter detections to keep only the highest-confidence bounding box per class.
        If multiple boxes exist for the same class, replace them with the next best candidate.
        """
        boxes_array = boxes.xyxy.cpu().numpy()  # Convert bounding box coordinates to NumPy array
        scores = boxes.conf.cpu().numpy()  # Confidence scores
        classes = boxes.cls.cpu().numpy().astype(int)  # Class IDs

        # Initialize dictionaries to store the best bounding boxes and backup candidates
        best_boxes = {}
        next_best_candidates = {}

        # Iterate through detections and store the highest confidence bounding box per class
        for i, cls in enumerate(classes):
            if cls not in best_boxes:
                # Store first detected bounding box for this class
                best_boxes[cls] = {'box': boxes_array[i], 'score': scores[i]}
            else:
                # Compare confidence scores and keep the higher confidence box
                if scores[i] > best_boxes[cls]['score']:
                    # Move the previous best box to the next-best candidate pool
                    if cls not in next_best_candidates:
                        next_best_candidates[cls] = best_boxes[cls]
                    else:
                        if best_boxes[cls]['score'] > next_best_candidates[cls]['score']:
                            next_best_candidates[cls] = best_boxes[cls]
                    
                    # Update best box with the new higher confidence detection
                    best_boxes[cls] = {'box': boxes_array[i], 'score': scores[i]}
                else:
                    # Store lower confidence box as a next-best candidate if none exists yet
                    if cls not in next_best_candidates or scores[i] > next_best_candidates[cls]['score']:
                        next_best_candidates[cls] = {'box': boxes_array[i], 'score': scores[i]}

        # Ensure detected keypoints are properly spaced and replace overlapping points
        filtered_boxes = []
        used_classes = set()

        for cls, data in best_boxes.items():
            box = data['box']
            if self.is_too_close(box, filtered_boxes, frame_shape):
                # Replace overlapping box with a next-best candidate if available
                if cls in next_best_candidates:
                    filtered_boxes.append(next_best_candidates[cls]['box'])
                    used_classes.add(cls)
                else:
                    print(f"Warning: No replacement found for class {cls}.")
            else:
                filtered_boxes.append(box)
                used_classes.add(cls)

        return filtered_boxes

    def is_too_close(self, box, filtered_boxes, frame_shape, threshold=50):
        """
        Check if a bounding box is too close to an already accepted box.
        If two bounding boxes are closer than the threshold, they might be duplicates.
        """
        x_min, y_min, x_max, y_max = box
        box_center = ((x_min + x_max) / 2, (y_min + y_max) / 2)  # Calculate center of the bounding box

        for fb in filtered_boxes:
            fb_x_min, fb_y_min, fb_x_max, fb_y_max = fb
            fb_center = ((fb_x_min + fb_x_max) / 2, (fb_y_min + fb_y_max) / 2)

            # Compute Euclidean distance between centers of bounding boxes
            distance = np.sqrt((box_center[0] - fb_center[0]) ** 2 + (box_center[1] - fb_center[1]) ** 2)
            if distance < threshold:  # If too close, consider it a duplicate
                return True
        return False

    def extract_bounding_box_centers(self, boxes):
        """Extract center points of bounding boxes to serve as keypoints for the court."""
        keypoints = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            x_center = int((x_min + x_max) / 2)
            y_center = int((y_min + y_max) / 2)
            keypoints.extend([x_center, y_center])  # Store as (x, y) coordinate pairs
        return keypoints

    def draw_keypoints(self, image, keypoints):
        """Draw the detected keypoints on the image for visualization."""
        for i in range(0, len(keypoints), 2):
            x, y = keypoints[i], keypoints[i + 1]
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Draw red circles at each keypoint
            cv2.putText(image, f'{i // 2}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label the keypoints
        return image
