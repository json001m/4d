import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5'


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mistral_inference.model import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))

# Initialize the distributed environment
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Define model paths
mistral_models_path = Path.home().joinpath(dir_path, 'mistral_models', '7B_instruct')

# Check if model files exist
def check_model_exists(model_path):
    required_files = ["params.json", "consolidated.safetensors", "tokenizer.model.v3"]
    for file in required_files:
        if not (model_path / file).exists():
            return False
    return True

# Main function to load the model and interact
def run(vrank, world_size):
    setup(vrank, world_size)
    print(f'rank1:{vrank}/{torch.distributed.get_rank()}')

    # Ensure the model directory exists
    if not check_model_exists(mistral_models_path):
        print("Model files do not exist. Please download and extract the model as per the instructions.")
        return

    # Load tokenizer and model using mistral_inference
    tokenizer = MistralTokenizer.from_file(str(mistral_models_path / "tokenizer.model.v3"))
    model = Transformer.from_folder(mistral_models_path, num_pipeline_ranks=6, device=vrank)
    #model.share_memory()

    print(f'({vrank}): from_folder()')

    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[vrank])

    print(f'({vrank}): ddp_model')


    # Define the system message
    system_message = UserMessage(content="You are a helpful pirate assistant that answers like a pirate.")

    # Start the CLI loop
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Prepare messages
        user_message = UserMessage(content=user_input)
        completion_request = ChatCompletionRequest(messages=[system_message, user_message])

        # Tokenize input
        tokens = tokenizer.encode_chat_completion(completion_request).tokens

        # Generate response
        out_tokens, _ = generate([tokens], ddp_model.module, max_tokens=64, temperature=0.7)
        response = tokenizer.decode(out_tokens[0])

        print("AI:", response)

if __name__ == "__main__":
    world_size = 6 #torch.cuda.device_count()
    torch.multiprocessing.spawn(run, args=(world_size,), nprocs=world_size, join=True)
