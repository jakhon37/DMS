import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)


class MyLogger(trt.ILogger):
    def __init__(self):
       trt.ILogger.__init__(self)

    def log(self, severity, msg):
        # pass # Your custom logging implementation here
        return msg 

logger = MyLogger()

builder = trt.Builder(logger)
# builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
# builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)

print(f'nework: {network}')

parser = trt.OnnxParser(network, logger)

model_path = 'weights/resnet100.onnx'
trt_path = 'weights/resnet100.engine'


success = parser.parse_from_file(model_path)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))

if not success:
    # pass # Error handling code here
    print(f'not succesfull: {success}')
else: print(f'succesfully loaded oonx model : {success}')

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB
print(f'all config set: {config}')
serialized_engine = builder.build_serialized_network(network, config)

with open(trt_path, 'wb') as f: 
    f.write(serialized_engine)
print(f'success')

# with open(“sample.engine”, “wb”) as f:
#     f.write(serialized_engine)