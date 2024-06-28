import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

def load_engine(trt_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(trt_path, 'rb') as f:
        engine_data = f.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream

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

def run_inference(engine, image_path):
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()
    input_shape = engine.get_binding_shape(0)
    
    input_tensor = preprocess_image(image_path, input_shape)
    np.copyto(inputs[0][0], input_tensor.ravel())
    
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
    
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    
    # Synchronize the stream
    stream.synchronize()
    
    return outputs[0][0]

def postprocess_output(output):
    bounding_boxes = output.reshape(-1, 2)
    return bounding_boxes

# Example usage:
trt_path = 'weights/adnet/train.engine'
image_path = 'data/indoor_029.png'

engine = load_engine(trt_path)
output = run_inference(engine, image_path)
bounding_boxes = postprocess_output(output)
print(bounding_boxes)
