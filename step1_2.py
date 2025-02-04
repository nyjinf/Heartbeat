# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:48:55 2024

@author: KF0032
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
# Define the GRU-based model
import torch
import torch.nn as nn
import torch.optim as optim

class SoftAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SoftAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_len, hidden_size)
        attention_scores = self.attention(hidden_states)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
        weighted_hidden_states = hidden_states * attention_weights  # (batch_size, seq_len, hidden_size)
        context_vector = weighted_hidden_states.sum(dim=1)  # (batch_size, hidden_size)
        return context_vector, attention_weights

class SoftGRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=False):
        super(SoftGRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.attention = SoftAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_size)
        context_vector, attention_weights = self.attention(gru_out)  # (batch_size, hidden_size)
        output = self.fc(context_vector)  # (batch_size, output_size)
        return output, attention_weights

# Example Usage
if __name__ == '__main__':
    batch_size = 32
    seq_len = 100
    input_size = 50
    hidden_size = 128
    output_size = 10
    num_layers = 1

    model = SoftGRUNet(input_size, hidden_size, output_size, num_layers)

    # Sample data
    inputs = torch.randn(batch_size, seq_len, input_size)  # Random input tensor
    outputs, attention_weights = model(inputs)


import os
import cv2
input_folder = 'UBSC/'
output_folder= 'Resized_Videos/'
new_width = 640
new_height = 480

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
subfolders= [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

for subfolder in subfolders:
    subfolder_path = os.path.join(input_folder, subfolder)
    output_subfolder = os.path.join(output_folder, subfolder)
    
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)
        
    video_files = [f for f in os.listdir(subfolder_path) if f.endswith(('.mp4','.avi'))]
    for video_file in video_files:
        video_path = os.path.join(subfolder_path, video_file)
        output_path = os.path.join(output_subfolder, f'resized_{video_file}')
    
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"error:could not open video file {video_file}")
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for output video
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (new_width, new_height))
            out.write(resized_frame)
        cap.release()   
        out.release()
        print(f"Resized video saved as {output_path}")
print("All videos resized")
            
input_base_path = 'Resized_Videos/'
output_base_path = 'stabilized_videos3/'


os.makedirs(output_base_path, exist_ok=True)


for folder in sorted(os.listdir(input_base_path)):
    folder_path = os.path.join(input_base_path, folder)
    
    if os.path.isdir(folder_path):
     
        for subfolder in sorted(os.listdir(folder_path)):
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                for video_file in sorted(os.listdir(subfolder_path)):
                    if video_file.endswith('.avi'):
                        input_video_path = os.path.join(subfolder_path, video_file)
                        output_video_path = os.path.join(output_base_path, f'stabilized_{video_file}')

                       
                        cap = cv2.VideoCapture(input_video_path)

                      
                        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)

                        
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

                       
                        _, prev = cap.read()
                        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

                        transforms = np.zeros((n_frames - 1, 3), np.float32)

                        for i in range(n_frames - 1):
                           
                            success, curr = cap.read()
                            if not success:
                                break

                           
                            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

                          
                            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

                            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

                        
                            good_prev_pts = prev_pts[status == 1]
                            good_curr_pts = curr_pts[status == 1]

                           
                            matrix = cv2.estimateAffinePartial2D(good_prev_pts, good_curr_pts)[0]

                           
                            dx = matrix[0, 2]
                            dy = matrix[1, 2]
                            da = np.arctan2(matrix[1, 0], matrix[0, 0])

                            # Store the transformations
                            transforms[i] = [dx, dy, da]

                            # Update the previous frame and points
                            prev_gray = curr_gray

                        # Apply stabilization by smoothing the trajectory
                        trajectory = np.cumsum(transforms, axis=0)

                       
                        window_size = 30
                        smoothed_trajectory = np.copy(trajectory)

                        for i in range(3):
                            smoothed_trajectory[:, i] = np.convolve(trajectory[:, i], np.ones(window_size) / window_size, mode='same')

                        # Compute the difference between smoothed and original trajectory
                        difference = smoothed_trajectory - trajectory

                        # Apply the reverse transformation to each frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        for i in range(n_frames - 1):
                            success, frame = cap.read()
                            if not success:
                                break

                            # Get the transform for this frame
                            dx, dy, da = transforms[i] + difference[i]

                            # Create transformation matrix
                            m = np.array([[np.cos(da), -np.sin(da), dx],
                                          [np.sin(da), np.cos(da), dy]])

                            # Apply the transformation
                            frame_stabilized = cv2.warpAffine(frame, m, (width, height))

                            # Write the stabilized frame
                            out.write(frame_stabilized)

                        # Release resources
                        cap.release()
                        out.release()

print("Frame Stabilization completed for all videos.")