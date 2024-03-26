import os
from dotenv import load_dotenv
try:
    from protoc.object_detection.object_detection_pb2_grpc import ObjectDetectionStub
    from protoc.object_detection.object_detection_pb2 import DetectionRequest, DetectionResponse
    from services.object_detection import parse_response as parse_object_detection_response
except:
    raise ImportError("Please compile proto files first!")


load_dotenv()

MAX_MESSAGE_LENGTH = int(os.getenv('MAX_MESSAGE_LENGTH', 4194304))
IMAGE_EXTENSIONS = ('.jpg', '.png', '.jpeg', '.webp')

SERVICE_MAP = {
    'object_detection': {
        'stub': ObjectDetectionStub,
        'message': (DetectionRequest, DetectionResponse),
        'parse_response': parse_object_detection_response,
    }
}