#!/bin/sh

python3 -m grpc_tools.protoc -I./protos --proto_path=./protos --python_out=./protos/python_out --pyi_out=./protos/python_out --grpc_python_out=./protos/python_out ./protos/agent_service.proto
sed "s/import agent_service_pb2 as agent__service__pb2/from \. import agent_service_pb2 as agent__service__pb2/g" ./protos/python_out/agent_service_pb2_grpc.py > ./protos/python_out/agent_service_pb2_grpc.py.tmp
mv -f ./protos/python_out/agent_service_pb2_grpc.py.tmp ./protos/python_out/agent_service_pb2_grpc.py
cp -f ./protos/python_out/* ./agent_server/src/protos/
cp -f ./protos/python_out/* ./agent_client_example/src/protos/