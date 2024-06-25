import grpc
import protos.agent_service_pb2_grpc as agent_service_pb2_grpc
import protos.agent_service_pb2 as agent_service_pb2

from concurrent import futures
import logging

from pprint import pp
from models import mistral

class AgentService(agent_service_pb2_grpc.AgentService):

    #right now, ignoring role and modelIdentifier parameters
    def queryModel(self, request: agent_service_pb2.ModelQueryRequest, context):
        print("QueryString: %s!" % request.queryString)
        response_text = mistral.query(request.queryString, request.systemMessage, request.temperature)
        response : agent_service_pb2.QueryResponse = agent_service_pb2.QueryResponse(response=response_text, requestId=request.id)
        print(response)
        return agent_service_pb2.QueryResponse(response=response.response)

def serve():
    # Set up GRPC service listener
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent_service_pb2_grpc.add_AgentServiceServicer_to_server(AgentService(), server)
    server.add_insecure_port("[::]:" + port)

    # listen for requests
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig()
    serve()
