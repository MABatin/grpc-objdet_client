import cv2
import numpy as np
import base64

def bytes_to_numpy(data: bytes):
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def encode_image(im_array: np.ndarray) -> str:
        """
        Encode numpy array to base64 string representation
        :param im_array:
        :return: str
        """
        # start = time.time()
        _, buffer = cv2.imencode('.jpg', im_array)
        base64_frame = base64.b64encode(buffer).decode('utf-8')

        return base64_frame


def image_to_bytes(image_path) -> bytes:
     with open(image_path, "rb") as image:
        f = image.read()
        return f