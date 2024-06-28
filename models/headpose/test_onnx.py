

import cv2
import numpy as np
import onnxruntime
import torch
import sys 
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'drepnet')))
from utils import draw_axis

# Function to preprocess the input image
def preprocess_image(image_path):
    im_cv2 = cv2.imread(image_path)
    image = cv2.cvtColor(im_cv2, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # Resize to the input size of the model
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = np.transpose(image, (2, 0, 1))  # Change data layout from HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return im_cv2, image

# Function to postprocess and draw Euler angles on the image
def draw_euler_angles(image_path, euler_angles):
    # image = cv2.imread(image_path)
    image = image_path
    pitch, yaw, roll = euler_angles[0]
    text = f'Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}'
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return image

# Load the ONNX model
onnx_model_path = 'weights/head_pose_model.onnx'
# onnx_model_path = 'weights/model.onnx'

session = onnxruntime.InferenceSession(onnx_model_path)

# Prepare input data
image_path = 'data/indoor_029.png'  # Replace with your image path
# image_path = 'data/300w/indoor_020.png'
im_cv2, input_image = preprocess_image(image_path)
input_name = session.get_inputs()[0].name

# Run inference
outputs = session.run(None, {input_name: input_image})
print(f'ouputs: {outputs[0][0]}')
euler_angles = outputs[0]
yaw, pitch, roll = euler_angles[0]
# Draw the results on the image
output_image = draw_euler_angles(im_cv2, euler_angles)
draw_axis(im_cv2, yaw, pitch, roll)

# Display the image with drawn Euler angles
# cv2.imshow('Output Image', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
base_name = os.path.basename(image_path)
# Save the image with drawn Euler angles
output_image_path = f'out_{base_name}'
cv2.imwrite(output_image_path, im_cv2)
