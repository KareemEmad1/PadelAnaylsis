from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utills import get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        # Run model on the frame to get tracking results
        results = self.model.track(frame, persist=True)

        id_name_dict = self.model.names
        player_detections = []

        for box in results[0].boxes:
            track_id = int(box.id.tolist()[0])  # Track ID of detected object
            coords = box.xyxy.tolist()[0]       # Bounding box coordinates
            object_cls_id = int(box.cls.tolist()[0])  # Class ID of detected object
            object_cls_name = id_name_dict[object_cls_id]
            confidence = float(box.conf.tolist()[0])  # Confidence score of the detection

            # Only add detections of persons to player_detections list
            if object_cls_name == "person":
                player_detections.append({
                    'track_id': track_id,
                    'coords': coords,
                    'confidence': confidence
                })

        # Sort players by confidence (highest first) and keep only top 4 players
        player_detections = sorted(player_detections, key=lambda x: x['confidence'], reverse=True)[:4]

        # Create a player_dict with only the top 4 players
        player_dict = {player['track_id']: player['coords'] for player in player_detections}

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                # Draw the bounding box and label for each player
                cv2.putText(frame, f"Player ID: {track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            output_video_frames.append(frame)

        return output_video_frames
