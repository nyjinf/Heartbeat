# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:49:52 2024

@author: KF0032
"""

import cv2
import numpy as np
import os


input_base_path = 'output_roi_face_video3/'  # Path to input videos  
output_base_path = 'output_skin_mask_video3/'  # Path to save output videos with skin masks


os.makedirs(output_base_path, exist_ok=True)

lower_skin = np.array([0, 20, 70], dtype=np.uint8)  
upper_skin = np.array([20, 255, 255], dtype=np.uint8)  # Upper HSV boundary for skin

# Iterate through each subfolder (e.g., s1 to s64)
for folder in sorted(os.listdir(input_base_path)):
    folder_path = os.path.join(input_base_path, folder)
    
    if os.path.isdir(folder_path):
        # Create a corresponding subfolder in the output path
        output_folder_path = os.path.join(output_base_path, folder)
        os.makedirs(output_folder_path, exist_ok=True)

        # Iterate through each video file in the subfolder
        for video_file in sorted(os.listdir(folder_path)):
            if video_file.endswith('.avi'):
                input_video_path = os.path.join(folder_path, video_file)
                output_video_path = os.path.join(output_folder_path, f'skin_mask_{video_file}')  # Add 'skin_mask_' prefix
                
                # Open the input video
                cap = cv2.VideoCapture(input_video_path)

                # Check if video opened successfully
                if not cap.isOpened():
                    print(f"Error opening video file: {video_file}")
                    continue

                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for saving the output video

                # Create VideoWriter object to save the video with skin mask
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)

                # Loop through the video frames
                while cap.isOpened():
                    ret, frame = cap.read()  # Read the next frame
                    if not ret:
                        break  # End of video

                    # Convert the frame to the HSV color space
                    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    # Create a binary mask where the skin colors fall within the boundaries
                    skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

                
                    kernel = np.ones((3, 3), np.uint8)
                    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
                    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_DILATE, kernel)

                    # Apply the mask to the original frame to extract the skin region
                    skin = cv2.bitwise_and(frame, frame, mask=skin_mask)

                    # Display the original frame and the skin mask (optional)
                    cv2.imshow('Original Frame', frame)
                    cv2.imshow('Skin Mask', skin)

                    # Write the frame with the skin mask to the output video
                    out.write(skin)

                   
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                out.release()
                cv2.destroyAllWindows()

print("Skin mask video processing complete!")
