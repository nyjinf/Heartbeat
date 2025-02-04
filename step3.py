# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:49:32 2024

@author: KF0032
"""

import cv2
import os

# Paths
input_base_path = 'Resized_Videos/'  
output_base_path = 'output_roi_face_video3/'  


os.makedirs(output_base_path, exist_ok=True)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


for folder in sorted(os.listdir(input_base_path)):
    folder_path = os.path.join(input_base_path, folder)

    if os.path.isdir(folder_path):
       
        output_folder_path = os.path.join(output_base_path, folder)
        os.makedirs(output_folder_path, exist_ok=True)

       
        for video_file in sorted(os.listdir(folder_path)):
            if video_file.endswith('.avi'):
                input_video_path = os.path.join(folder_path, video_file)
                output_video_path = os.path.join(output_folder_path, f'roi_{video_file}')  # Add 'roi_' prefix to output file name

            
                cap = cv2.VideoCapture(input_video_path)

            
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video codec
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            
                while cap.isOpened():
                    ret, frame = cap.read() 
                    if not ret:
                        break  

               
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

              
                    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                   
                    for (x, y, w, h) in faces:
                        
                        face_roi = frame[y:y+h, x:x+w]

                   
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Save the frame with the drawn face rectangle to the output video
                    out.write(frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

          
                cap.release()
                out.release()

print("Face ROI extraction and video processing complete!")
