python -m grpc_tools.protoc -Iprotoc/$1=proto \
 --python_out=. --pyi_out=. --grpc_python_out=. proto/$1.proto