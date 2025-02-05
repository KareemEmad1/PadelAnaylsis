from ultralytics import YOLO
import os
import pickle
import cv2
import pandas as pd
import numpy as np

class BallTracker:
    def __init__(self, model_path, max_missing_frames=10):
        """
        Initializes the BallTracker class with a YOLO model for detecting the ball.
        
        Parameters:
        - model_path (str): Path to the trained YOLO model for ball detection.
        - max_missing_frames (int): Maximum consecutive frames where the ball can be missing before interpolation.
        """
        self.model = YOLO(model_path)  # Load the YOLO model for ball detection
        self.max_missing_frames = max_missing_frames  # Define threshold for missing frames before interpolation

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolates missing ball positions to ensure smooth tracking when the ball disappears for some frames.
        
        Parameters:
        - ball_positions (list): List of dictionaries containing ball bounding box coordinates.

        Returns:
        - List of dictionaries with interpolated ball positions.
        """
        ball_positions = [x.get(1, []) for x in ball_positions]  # Extract the ball's bounding boxes

        # Convert the list into a Pandas DataFrame for easy handling of missing values
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values (fills in gaps where the ball disappears temporarily)
        df_ball_positions = df_ball_positions.interpolate()  # Interpolates missing values linearly
        df_ball_positions = df_ball_positions.bfill()  # Backfills any remaining missing values

        # Convert back to list format for further processing
        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detects the ball in each frame of the video.
        
        Parameters:
        - frames (list): List of frames from the input video.
        - read_from_stub (bool): If True, load precomputed detections from a stub file instead of running detection.
        - stub_path (str): Path to save or load precomputed detections.

        Returns:
        - List of dictionaries containing ball detections for each frame.
        """
        ball_detections = []

        # If a precomputed detection file (stub) exists, load it to save computation time
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        # Process each frame and detect the ball
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        # Save detections to a stub file for future use
        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        """
        Runs YOLO object detection on a single frame to detect the ball.

        Parameters:
        - frame (numpy array): The current frame from the video.

        Returns:
        - Dictionary containing the bounding box of the detected ball.
        """
        results = self.model.predict(frame, conf=0.15)[0]  # Run YOLO model with a confidence threshold of 15%
        ball_dict = {}  # Dictionary to store detected ball position
        frame_width, frame_height = 1920, 1080  # Assume fixed video resolution (can be made dynamic)

        # Loop through detected objects in the frame
        for box in results.boxes:
            coords = box.xyxy.tolist()[0]  # Convert detection coordinates to a list
            x1, y1, x2, y2 = coords  # Extract bounding box coordinates

            # Ensure the detected object is within the frame boundaries
            if 0 <= x1 < frame_width and 0 <= y1 < frame_height and 0 <= x2 < frame_width and 0 <= y2 < frame_height:
                ball_dict[1] = coords  # Assign detected ball's bounding box to dictionary

        return ball_dict  # Return detected ball's position (or empty if no ball detected)

    def draw_bboxes(self, video_frames, ball_detections):
        """
        Draws bounding boxes around the detected ball in each frame.

        Parameters:
        - video_frames (list): List of frames from the video.
        - ball_detections (list): List of dictionaries containing ball bounding box coordinates.

        Returns:
        - List of frames with drawn bounding boxes.
        """
        output_video_frames = []

        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox  # Extract bounding box coordinates
                
                # Draw a yellow bounding box around the detected ball
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

            output_video_frames.append(frame)

        return output_video_frames  # Return frames with drawn bounding boxes
