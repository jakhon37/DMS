import cv2
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from six import string_types

# Preprocessing function
def preprocess_image(image_path, input_shape):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_shape[2], input_shape[3]))
    image = image.astype(np.float32)
    image = image / 255.0  # Normalizing to range [0, 1]
    image = (image - 0.5) / 0.5  # If the model requires mean subtraction and division
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return np.ascontiguousarray(image)

# Existing Binding and Engine classes
class Binding(object):
    def __init__(self, engine, idx_or_name):
        if isinstance(idx_or_name, string_types):
            self.name = idx_or_name
        else:
            self.index = idx_or_name
            self.name = engine.get_tensor_name(self.index)
            if self.name is None:
                raise IndexError("Binding index out of range: %i" % self.index)
        self.is_input = engine.get_tensor_mode(self.name) == trt.TensorIOMode.INPUT

        dtype = engine.get_tensor_dtype(self.name)
        dtype_map = {trt.DataType.FLOAT: np.float32,
                     trt.DataType.HALF: np.float16,
                     trt.DataType.INT8: np.int8,
                     trt.DataType.BOOL: np.bool_}
        if hasattr(trt.DataType, 'INT32'):
            dtype_map[trt.DataType.INT32] = np.int32
        if hasattr(trt.DataType, 'INT64'):
            dtype_map[trt.DataType.INT64] = np.int64

        self.dtype = dtype_map[dtype]
        shape = engine.get_tensor_shape(self.name)
        self.shape = tuple(shape)
        self._host_buf = None
        self._device_buf = None

    @property
    def host_buffer(self):
        if self._host_buf is None:
            self._host_buf = cuda.pagelocked_empty(self.shape, self.dtype)
        return self._host_buf

    @property
    def device_buffer(self):
        if self._device_buf is None:
            self._device_buf = cuda.mem_alloc(self.host_buffer.nbytes)
        return self._device_buf

    def get_async(self, stream):
        cuda.memcpy_dtoh_async(self.host_buffer, self.device_buffer, stream)
        return self.host_buffer

def check_input_validity(input_idx, input_array, input_binding):
    trt_shape = tuple(input_binding.shape)
    onnx_shape = tuple(input_array.shape)

    if onnx_shape != trt_shape:
        if not (trt_shape == (1,) and onnx_shape == ()):
            raise ValueError("Wrong shape for input %i. Expected %s, got %s." %
                             (input_idx, trt_shape, onnx_shape))

    if input_array.dtype != input_binding.dtype:
        if input_array.dtype == np.int64 and input_binding.dtype == np.int32:
            casted_input_array = np.array(input_array, copy=True, dtype=np.int32)
            if np.equal(input_array, casted_input_array).all():
                input_array = casted_input_array
            else:
                raise TypeError("Wrong dtype for input %i. Expected %s, got %s. Cannot safely cast." %
                                (input_idx, input_binding.dtype, input_array.dtype))
        else:
            raise TypeError("Wrong dtype for input %i. Expected %s, got %s." %
                            (input_idx, input_binding.dtype, input_array.dtype))
    return input_array

class Engine(object):
    def __init__(self, trt_engine):
        self.engine = trt_engine

        bindings = [Binding(self.engine, i)
                    for i in range(self.engine.num_io_tensors)]
        self.binding_addrs = [int(b.device_buffer) for b in bindings]
        self.inputs = [b for b in bindings if b.is_input]
        self.outputs = [b for b in bindings if not b.is_input]

        for binding in self.inputs + self.outputs:
            _ = binding.device_buffer  # Force buffer allocation
        for binding in self.outputs:
            _ = binding.host_buffer  # Force buffer allocation
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

    def run(self, inputs):
        if len(inputs) < len(self.inputs):
            raise ValueError("Not enough inputs. Expected %i, got %i." %
                             (len(self.inputs), len(inputs)))
        if isinstance(inputs, dict):
            inputs = [inputs[b.name] for b in self.inputs]

        for i, (input_array, input_binding) in enumerate(zip(inputs, self.inputs)):
            input_array = check_input_validity(i, input_array, input_binding)
            cuda.memcpy_htod_async(input_binding.device_buffer, input_array, self.stream)

        num_io = self.engine.num_io_tensors
        for i in range(num_io):
            tensor_name = self.engine.get_tensor_name(i)
            if i < len(inputs) and self.engine.is_shape_inference_io(tensor_name):
                self.context.set_tensor_address(tensor_name, inputs[i].ctypes.data)
            else:
                self.context.set_tensor_address(tensor_name, self.binding_addrs[i])

        self.context.execute_async_v3(self.stream.handle)
        results = [output.get_async(self.stream)
                   for output in self.outputs]
        self.stream.synchronize()
        return results

    def run_no_dma(self):
        self.context.execute_async_v3(self.stream.handle)

# Main script
if __name__ == "__main__":
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    trt_path = 'weights/resnet100.engine'
    with open(trt_path, 'rb') as f:
        serialized_engine = f.read()
    print('Loaded the TRT engine.')

    engine = runtime.deserialize_cuda_engine(serialized_engine)
    print('Engine is ready.')

    input_shape = (1, 3, 112, 112)
    engine_wrapper = Engine(engine)

    # Path to your image
    image_path = 'path_to_your_image.jpg'
    image_path = 'data/indoor_029.png'
    
    h_input = preprocess_image(image_path, input_shape)

    # Run inference
    h_output = engine_wrapper.run([h_input])

    print("Inference output:", h_output)


    # h_input = torch.randn(input_shape).numpy().astype(np.float32)
    
    # image_path = 'data/indoor_029.png'
    # h_input = preprocess_image(image_path, input_shape)
    
    # h_output = engine_wrapper.run([h_input])

    # print("Inference output:", h_output)
