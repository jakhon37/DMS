

onnx_file_path = 'weights/resnet100.onnx'

    
# python
import os
import tensorrt as trt
batch_size = 1
TRT_LOGGER = trt.Logger()
def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # builder.max_workspace_size = 1 << 30
        # builder.max_batch_size = batch_size
        builder 
        # Load the Onnx model and parse it in order to populate the TensorRT network.        
    with open(model_file, 'rb') as model:
        parser.parse(model.read())
    return network
    # return builder.build_cuda_engine(network)
        
# downloaded the arcface mdoel
# onnx_file_path = './resnet100.onnx'
    
engine = build_engine_onnx(onnx_file_path)
engine_file_path = './arcface_trt.engine'
with open(engine_file_path, "wb") as f:
    f.write(engine.serialize())
