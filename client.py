import os
from typing import Iterator, Union, Callable
from utils.logger import logger
from utils.helper import bytes_to_numpy
import argparse
import grpc
import cv2
from concurrent.futures import ThreadPoolExecutor
import threading
from config import MAX_MESSAGE_LENGTH, SERVICE_MAP, IMAGE_EXTENSIONS

quit_view = False

def response_watcher(response_iterator: Iterator, parse_response: Callable,
                     video_writer: cv2.VideoWriter = None, input_type: str = "video",
                     image_lock: threading.Lock = None, window_name: str = ""):
        global quit_view

        for response in response_iterator:
            try:
                json_response = parse_response(response)

                logger.debug(f"Frame:{json_response['frame_id']} num objects: {len(json_response['results'])}")
                logger.debug(f"Frame:{json_response['frame_id']} results: {json_response['results']}")

                plot_im = bytes_to_numpy(response.image)
                if video_writer is not None:  
                    video_writer.write(plot_im)
                if image_lock is not None and not quit_view:
                    with image_lock:
                        cv2.imshow(window_name, plot_im)                                        
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    if input_type == "video":
                        logger.info("========== Quit view ==========")
                        logger.info("===== video is being saved in the background =====")
                        quit_view = True
                    if input_type == "webcam":
                        logger.info("========== Quit program ==========")
                        os._exit(0)
                    
            except Exception as e:
                 logger.error(f"Error in response: {e}")


def run_video(client, Request, input_file: Union[str, int], parse_response: Callable,
              output_file =  None, desired_fps: int = 1, classes: list[int] = None, 
              show = False):
    global quit_view

    executor = ThreadPoolExecutor(max_workers=1)
    window_name = f"{type(client).__name__} View"
    image_lock = None
    if show and not quit_view:
        image_lock = threading.Lock()
    input_type = "webcam" if isinstance(input_file, int) else "video"

    # Load the video
    video = cv2.VideoCapture(input_file)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_file is not None:
        output_video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), desired_fps, (frame_width, frame_height))
    else:
         output_video = None

    if not video.isOpened():
        logger.error("Unable to open video file")
        exit(1)
        
    frame_idx = 0
    while True:
        try:
            ret, frame = video.read()

            if not ret:
                break

            im = cv2.imencode('.jpg', frame)[1].tobytes()
            # TODO: check for better approach here
            if not frame_idx % (fps//desired_fps) == 0:
                frame_idx += 1
                continue

            request = Request(
                frame_id=frame_idx,
                classes=classes,
                image=im
            )
            frame_idx += 1

            response_iterator = client.DetectStream(iter((request,)))
            response_future = executor.submit(
                response_watcher, response_iterator, 
                parse_response, output_video, input_type, 
                image_lock, window_name
            )
            if response_future.done():
                output_video.release()
        except KeyboardInterrupt:
            video.release()
            cv2.destroyAllWindows()
            logger.info("========== Quit program ==========")
            os._exit(0)


    # Release the video and close the window
    video.release()
    cv2.destroyAllWindows()


def run_image(client, Request, input_file: str, parse_response: Callable,
              output_file =  None, 
              classes: list[int] = None, show=False):
    im = cv2.imread(input_file)
    im = cv2.imencode('.jpg', im)[1].tobytes()

    request = Request(
        frame_id=0,
        classes=classes,
        image=im
    )
    response = client.DetectImage(request)
    json_response = parse_response(response)

    logger.debug(f"Num objects: {len(json_response['results'])}")
    logger.debug(f"Detection results: {json_response['results']}")

    if output_file is not None:
        cv2.imwrite(output_file, bytes_to_numpy(response.image))
    if show:
        cv2.imshow("Detection", bytes_to_numpy(response.image))
        cv2.waitKey(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", type=str, default="object_detection", help="Service task")
    parser.add_argument("--classes", nargs='+', default=None, help="Classes to detect")
    parser.add_argument("--host", type=str, default="localhost", help="Server host address")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output video file")
    parser.add_argument("--out-fps", type=int, default=1, help="Output FPS")
    parser.add_argument("--show", action="store_true", help="Show detection image")

    return parser.parse_args()    


if __name__ == "__main__":
    args = parse_args()

    if args.classes is not None:
        try:
            args.classes = [int(x) for x in args.classes]
        except:
            raise ValueError("Please provide valid classes (int)!")
    
    channel = grpc.insecure_channel(
        f"{args.host}:{args.port}",
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
        ]
    )
    stub = SERVICE_MAP[args.service]['stub']
    RequestType, ResponseType = SERVICE_MAP[args.service]['message']
    parse_response_func = SERVICE_MAP[args.service]['parse_response']

    client = stub(channel)

    if os.path.splitext(args.input)[1] in IMAGE_EXTENSIONS:
        run_image(client=client, Request=RequestType, parse_response=parse_response_func,
            input_file=args.input, output_file=args.output, show=args.show, classes=args.classes)
    else:
        # TODO: figure out showing images from response watcher thread
            # raise NotImplementedError("Showing detection video currently unsupported!")
        
        if args.input == "0":
            args.input = int(args.input)  # Webcam                
            if args.show:
                logger.info("Press 'q' to quit view and program...")
        else:
            if args.show:
                logger.info("Press 'q' to quit view (ctrc+c to quit program)...")

        run_video(client=client, Request=RequestType, parse_response=parse_response_func,
            input_file=args.input, output_file=args.output, classes=args.classes,
            desired_fps=args.out_fps, show=args.show)