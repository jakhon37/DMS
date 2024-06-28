import onnxruntime as ort
import cv2
import numpy as np

def get_model_input_shape(model_path):
    session = ort.InferenceSession(model_path)
    input_shape = session.get_inputs()[0].shape
    input_name = session.get_inputs()[0].name
    return input_shape, input_name

def preprocess_image(image_path, input_shape):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (input_shape[2], input_shape[3]))
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Ensuring the image values are in the range [0, 1]
    image_normalized = np.clip(image_normalized, 0, 1)
    
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    input_tensor = np.expand_dims(image_transposed, axis=0)
    
    # Ensure the input tensor is of type float32
    input_tensor = input_tensor.astype(np.float32)
    
    return input_tensor

def run_inference(model_path, image_path):
    session = ort.InferenceSession(model_path)
    input_shape, input_name = get_model_input_shape(model_path)
    print(f'Model input shape: {input_shape}')
    print(f'Model input name: {input_name}')
    
    input_tensor = preprocess_image(image_path, input_shape)
    print(f'Input tensor shape: {input_tensor.shape}')
    print(f'Input tensor dtype: {input_tensor.dtype}')
    print(f'Input tensor min value: {input_tensor.min()}')
    print(f'Input tensor max value: {input_tensor.max()}')
    
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_tensor})
    return outputs

def postprocess_output(outputs):
    bounding_boxes = outputs[0]
    return bounding_boxes

def draw_landmarks(image_path, landmarks):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    landmarks = landmarks[0]  # Removing the batch dimension
    landmarks[:, 0] = (landmarks[:, 0] + 1) / 2 * width  # Normalize x to [0, width]
    landmarks[:, 1] = (landmarks[:, 1] + 1) / 2 * height  # Normalize y to [0, height]
    for (x, y) in landmarks:
        center = (int(x), int(y))
        cv2.circle(image, center, 2, (0, 255, 0), -1)
    return image

# Example usage:
# model_path = 'weights/2d106det.onnx'
# model_path = 'weights/1k3d68.onnx'
# model_path = 'weights/adnet/train.onnx'
model_path = 'weights/adnet/300w.onnx'


image_path = 'data/indoor_021.png'
outputs = run_inference(model_path, image_path)
print(f'outputs len: {len(outputs)}')
# print(f'outputs: {outputs}')

print(f'outputs[0] shape: {outputs[0].shape}')
print(f'outputs[0]  dtype: {outputs[0].dtype}')

im = draw_landmarks(image_path, landmarks=outputs[0])
cv2.imwrite('imonnx.jpg', im)
bounding_boxes = postprocess_output(outputs)
print(f'len output : {len(bounding_boxes)}')
# print(bounding_boxes)
