import argparse
import arrow
import pycuda.driver as cuda  # noqa, must be imported
import pycuda.autoinit  # noqa, must be imported
import tensorrt as trt
import numpy as np
from bistiming import Stopwatch
from eyewitness.detection_utils import DetectionResult
from eyewitness.image_id import ImageId
from eyewitness.config import BoundedBoxObject
from eyewitness.object_detector import ObjectDetector
from eyewitness.image_utils import ImageHandler, Image, resize_and_stack_image_objs
import common

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# class YOLO defines the default value, so suppress any default here
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
'''
Command line options
'''
parser.add_argument(
    '--engine_file', type=str, help='path to model weight file'
)


class TensorRTArcFaceDetectorWrapper(ObjectDetector):
    def __init__(self, engine_file, image_shape=(112, 112)):
        self.image_shape = image_shape  # (H, W)
        self.engine_file = engine_file
        self.engine = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        self.context = None

    def build(self):
        with open(self.engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()

    def detect(self, image_obj) -> DetectionResult:
        # lazy load implementation
        if self.engine is None:
            self.build()

        image_raw_width = image_obj.pil_image_obj.width
        image_raw_height = image_obj.pil_image_obj.height

        self.inputs[0].host = self.preprocess(image_obj.pil_image_obj)

        trt_outputs = common.do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs,
            stream=self.stream)

        features = trt_outputs[0]  # assuming the output is the feature vector

        detected_objects = [BoundedBoxObject(0, 0, image_raw_width, image_raw_height, "face", 1.0, features)]

        image_dict = {
            'image_id': image_obj.image_id,
            'detected_objects': detected_objects,
        }
        detection_result = DetectionResult(image_dict)
        return detection_result

    def preprocess(self, pil_image_obj):
        """
        Preprocess the input image to match the input shape required by the ArcFace model
        """
        image = pil_image_obj.resize(self.image_shape)
        image = np.array(image, dtype=np.float32)
        image = image / 255.0  # normalize to range [0, 1]
        image = np.transpose(image, (2, 0, 1))  # change data layout from HWC to CHW
        image = np.expand_dims(image, axis=0)  # add batch dimension
        return np.ascontiguousarray(image)  # make sure the array is contiguous


if __name__ == '__main__':
    model_config = parser.parse_args()
    
    model_config.engine_file = 'data/indoor_029.png'
    
    object_detector = TensorRTArcFaceDetectorWrapper(model_config.engine_file)
    raw_image_path = 'demo/test_image.jpg'
    image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp(), file_format='jpg')
    image_obj = Image(image_id, raw_image_path=raw_image_path)
    with Stopwatch('Running inference on image {}...'.format(raw_image_path)):
        detection_result = object_detector.detect(image_obj)
    ImageHandler.draw_bbox(image_obj.pil_image_obj, detection_result.detected_objects)
    ImageHandler.save(image_obj.pil_image_obj, "detected_image/drawn_image.jpg")
