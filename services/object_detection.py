from protoc.object_detection.object_detection_pb2 import DetectionResponse
from collections import defaultdict

def parse_response(response: DetectionResponse):
    """
    Parse the response from the gRPC server into JSONable object
    """
    json_response = defaultdict()
    json_response['frame_id'] = response.frame_id
    json_response['results'] = []
    for result in response.results:
        json_result = defaultdict()
        json_result['label'] = result.label
        json_result['score'] = result.score
        json_result['xyxy'] = result.xyxy
        json_result['xyxyn'] = result.xyxyn
        json_result['xywh'] = result.xywh
        json_result['xywhn'] = result.xywhn
        json_response['results'].append(json_result)
    
    return json_response