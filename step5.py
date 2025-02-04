# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:50:20 2024

@author: KF0032
"""

#Individual Typology Angle (ITA) estimation with Contrast Limited Adaptive Histogram Equalization (CLAHE)
import cv2
import numpy as np
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Function to apply CLAHE to an image and calculate ITA
def calculate_ita_with_clahe(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into L*, a*, and b* channels
    L, a, b = cv2.split(lab_image)
    
    # Apply CLAHE to the L* channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)
    
    # Merge the enhanced L* channel back with a* and b*
    lab_image_clahe = cv2.merge([L_clahe, a, b])
    
    # Convert back to BGR to visualize the enhanced image
    enhanced_image = cv2.cvtColor(lab_image_clahe, cv2.COLOR_LAB2BGR)
    
    # Calculate ITA based on L* and b* channels
    L_clahe = L_clahe.astype(np.float32)
    b = b.astype(np.float32)
    
    # Prevent division by zero by setting b values of 0 to a small value
    b[b == 0] = 1e-6
    
    # Calculate ITA in degrees
    ita = np.arctan2(L_clahe - 50, b) * 180 / np.pi
    
    # Return the enhanced image and ITA
    return enhanced_image, ita



import cv2
import numpy as np
import os

# Input and output video paths
input_base_path = 'output_skin_mask_video3/'  # Path to input videos  
output_base_path = 'enhanced_skin_video3/'  # Path to save output videos

# Create the output directory if it doesn't exist
os.makedirs(output_base_path, exist_ok=True)

# Define the lower and upper boundaries for skin color in HSV
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Iterate through each subfolder in the input directory (e.g., s1 to s64)
for folder in sorted(os.listdir(input_base_path)):
    folder_path = os.path.join(input_base_path, folder)
    
    if os.path.isdir(folder_path):
        # Create a corresponding subfolder in the output directory
        output_folder_path = os.path.join(output_base_path, folder)
        os.makedirs(output_folder_path, exist_ok=True)

        # Iterate through each video file in the subfolder
        for video_file in sorted(os.listdir(folder_path)):
            if video_file.endswith('.avi'):
                input_video_path = os.path.join(folder_path, video_file)
                output_video_path = os.path.join(output_folder_path, f'enhanced_{video_file}')  # Add 'enhanced_' prefix
                
                # Open the input video
                cap = cv2.VideoCapture(input_video_path)

                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for saving the output video

                # Create VideoWriter object to save the enhanced skin video
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)

                # Loop through the video frames
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break  # End of video

                    # Convert the frame to HSV color space
                    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    # Create a binary mask for skin color
                    skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

                    # Enhance the skin color by applying the mask
                    skin_image = cv2.bitwise_and(frame, frame, mask=skin_mask)

                    # Convert the skin-enhanced image to LAB color space for better color manipulation
                    lab_image = cv2.cvtColor(skin_image, cv2.COLOR_BGR2Lab)

                    # Split the LAB image into L, A, and B channels
                    l, a, b = cv2.split(lab_image)

                    # Enhance contrast and brightness by adjusting the L channel
                    l = cv2.equalizeHist(l)  # Histogram equalization to enhance contrast

                    # Merge the enhanced L channel with the original A and B channels
                    enhanced_lab_image = cv2.merge([l, a, b])

                    # Convert back to BGR color space
                    enhanced_frame = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_Lab2BGR)

                    # Write the enhanced frame to the output video
                    out.write(enhanced_frame)

                    # Display the original and enhanced frames (optional)
                    cv2.imshow('Original Frame', frame)
                    cv2.imshow('Enhanced Frame', enhanced_frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Release resources for this video
                cap.release()
                out.release()
                cv2.destroyAllWindows()

print("Skin color enhancement for videos complete!")
