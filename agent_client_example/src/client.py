import logging

import grpc
import protos.agent_service_pb2 as agent_service_pb2
import protos.agent_service_pb2_grpc as agent_service_pb2_grpc

def run(prompt, system_message, temperature):
    print("Requesting...")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = agent_service_pb2_grpc.AgentServiceStub(channel)
        response = stub.queryModel(agent_service_pb2.ModelQueryRequest(modelIdentifier="mistral", queryString=prompt, systemMessage=system_message, temperature=temperature, id=0))
    print("received: " + response.response)

if __name__ == "__main__":
    logging.basicConfig()
    prompt = "ocean"
    system_message = "You are a helpful assistant. You will respond to all requests with one word only, and that word will be the color best associated with the query. You will not provide any response beyond this single word."
    temperature = 0.7
    run(prompt, system_message, temperature)
