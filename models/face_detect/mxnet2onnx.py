
#  numpy 1.19.0

import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet

def convert_mxnet_to_onnx(prefix, epoch, input_shape, onnx_file_path):
    # Load the MXNet model
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    
    # Convert the MXNet model to ONNX format
    converted_model_path = onnx_mxnet.export_model(
        sym=sym, 
        params=[arg_params, aux_params], 
        input_shape=input_shape, 
        input_type='float32', 
        onnx_file_path=onnx_file_path
    )
    return converted_model_path

# Example usage
prefix = 'weights/retinaface-R50/R50'  # Path to the .json file without the extension
epoch = 0  # Epoch number of the model
input_shape = [(1, 3, 112, 112)]  # Replace with your model's input shape
onnx_file_path = 'retinaface-R50.onnx'

converted_model_path = convert_mxnet_to_onnx(prefix, epoch, input_shape, onnx_file_path)
print(f'Model converted to ONNX format and saved at {converted_model_path}')


# AttributeError: No conversion function registered for op type SoftmaxActivation yet.