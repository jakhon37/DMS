from models.face_detection import FaceDetectionModel
from models.eye_detection import EyeDetectionModel
from models.eye_gaze_detection import EyeGazeDetectionModel
from models.object_detection import ObjectDetectionModel
from utils.image_processing import preprocess_image, draw_bounding_boxes
from utils.video_stream import VideoStream
from config.config import Config

def main():
    # Load models
    face_model = FaceDetectionModel(Config.FACE_DETECTION_MODEL_PATH)
    eye_model = EyeDetectionModel(Config.EYE_DETECTION_MODEL_PATH)
    gaze_model = EyeGazeDetectionModel(Config.EYE_GAZE_DETECTION_MODEL_PATH)
    object_model = ObjectDetectionModel(Config.OBJECT_DETECTION_MODEL_PATH)

    # Start video stream
    video_stream = VideoStream(Config.VIDEO_SOURCE)
    
    while True:
        frame = video_stream.read_frame()
        if frame is None:
            break
        
        # Preprocess the frame
        processed_frame = preprocess_image(frame)
        
        # Detect faces
        faces = face_model.detect_faces(processed_frame)
        
        for face in faces:
            # Detect eyes in the face region
            eyes = eye_model.detect_eyes(face)
            for eye in eyes:
                # Detect eye gaze
                gaze = gaze_model.detect_eye_gaze(eye)
                # Use gaze data as needed
            
            # Detect objects (e.g., cigarette, phone, cap)
            objects = object_model.detect_objects(processed_frame)
        
        # Draw bounding boxes and labels
        draw_bounding_boxes(frame, faces, [FACE_LABEL] * len(faces))
        draw_bounding_boxes(frame, objects, OBJECT_LABELS)
        
        # Display the frame
        cv2.imshow('Driver Monitoring System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_stream.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
