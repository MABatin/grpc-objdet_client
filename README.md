# gRPC Object Detection Client
A simple gRPC Python client for communicating with gRPC server for various object detection service. Supports both image and video input, as well as saving detection results in an output video file.

# Installation

--------------------------------
**Use Python version 3.10**

## Conda Environment
Make sure to have _anaconda_ or _miniconda_ installed

1. `conda create -n grpcenv python=3.10`
2. `conda activate grpcenv`
3. `cd grpc-ml-server`
4. `pip install -r requirements.txt`

## Virtualenv

1. `cd grpc-ml-server`
2. `pip3 install virtualenv` (if virtualenv is not installed)
3. `python3 -m virtualenv venv`
4. `source venv/bin/activate`
5. `pip install -r requirements.txt`


## Compile proto file
When accessing a new service, proto file should be compiled with the following command:\
`bash compile_proto.sh [SERVICE_NAME]`

# Running the client
-----
`python client.py --host <HOST> --port <PORT> --input <PATH_TO_INPUT>`\
[OPTIONAL]  `--output <PATH_TO_OUTPUT> --out-fps <FPS> --classes <LIST_OF_CLASSES> --show`


# TODO
---
- [ ] Add support for sending batch of images 
- [ ] Add support for logging to file
- [ ] Add environment variables
- [ ] Add docker support for deployment


## Future plans for improvements
* Process response with detection metadata for further functionality
* Download .proto files of a service directly from server
* Implement queue for improved performance (maybe)
  
**Contributions are welcome 😃**