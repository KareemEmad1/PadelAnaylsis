import os
import cv2
import torch
import pandas as pd
import numpy as np
from utills import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_detector import CourtLineDetector


def main():

    # Define the input video path
    input_video_path = "input/test_keypoints.mp4"

    # Read video frames into a list
    video_frames = read_video(input_video_path)

    # Check if video frames were successfully read
    if not video_frames:
        print("Error: No frames were read from the video.")
        return

    # Initialize YOLO-based player tracker
    player_tracker = PlayerTracker(model_path="best_players.pt")

    # Initialize YOLO-based ball tracker
    ball_tracker = BallTracker(model_path="models/best_ball.pt")

    # Initialize YOLO-based court detector
    court_detector = CourtLineDetector(model_path="models/best_court.pt")

    # Detect court keypoints from the first frame of the video
    print("Detecting court keypoints from the first frame...")
    first_frame = video_frames[0]
    court_keypoints = court_detector.predict(first_frame)

    # Overlay detected court keypoints on all video frames
    video_frames_with_court = [
        court_detector.draw_keypoints(frame.copy(), court_keypoints) for frame in video_frames
    ]

    # Detect players across all frames (using stub data if available)
    print("Detecting players...")
    player_detections = player_tracker.detect_frames(
        video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections_test4.pkl"
    )

    # Detect the ball across all frames (does not use stub data)
    print("Detecting ball...")
    ball_detections = ball_tracker.detect_frames(
        video_frames, read_from_stub=False, stub_path="tracker_stubs/ball_detections_test_keypoints.pkl"
    )

    # Initialize lists to store detected ball coordinates
    x_ball, y_ball = [], []

    # Process each frame's ball detection results
    for i, frame_data in enumerate(ball_detections):
        if 1 in frame_data:  # If the ball is detected in this frame
            bbox = frame_data[1]
            cx, cy = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)  # Compute the center of the bounding box
            x_ball.append(cx)
            y_ball.append(cy)
        else:
            x_ball.append(np.nan)  # No ball detected, store NaN
            y_ball.append(np.nan)  # No ball detected, store NaN

    # Create a DataFrame to store ball position data
    ball_data = pd.DataFrame({'frame': list(range(len(video_frames))), 'x': x_ball, 'y': y_ball})

    # Interpolate missing values for ball positions to ensure continuous tracking
    ball_data['x'] = ball_data['x'].interpolate(limit_direction='both')
    ball_data['y'] = ball_data['y'].interpolate(limit_direction='both')

    # Calculate the velocity of the ball using the Euclidean distance formula
    ball_data['V'] = np.sqrt(ball_data['x'].diff()**2 + ball_data['y'].diff()**2)

    # Initialize a bounce column with 0 (indicating no bounce)
    ball_data['bounce'] = 0  

    # Ensure the training data folder exists
    os.makedirs('training_data', exist_ok=True)

    # Save ball position and movement data for future training of the bounce detection model
    csv_path = 'training_data/ball_positions_test_keypoints.csv'
    ball_data.to_csv(csv_path, index=False)
    print(f"Ball positions saved to {csv_path}")

    # Draw bounding boxes for detected players on the video frames
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    # Draw bounding boxes for detected balls on the video frames
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # Annotate each frame with a frame counter and court keypoints
    for i in range(len(output_video_frames)):
        # Display the frame number on the video
        frame_count_text = f"Frame: {i}"
        cv2.putText(output_video_frames[i], frame_count_text, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Overlay detected court keypoints on the frame
        output_video_frames[i] = court_detector.draw_keypoints(output_video_frames[i], court_keypoints)

    # Ensure the output video folder exists
    print("Saving the output video...")
    os.makedirs("output_videos", exist_ok=True)

    # Save the processed video with annotations
    save_video(output_video_frames, "output_videos/test_keypoints_output.avi")
    print("Video saved successfully!")


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
