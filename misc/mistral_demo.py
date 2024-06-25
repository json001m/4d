import json
import struct
from pprint import pp
from mistral_inference.model import Transformer, ModelArgs
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_inference.generate import generate
from accelerate import Accelerator
import torch
from torch import nn, distributed as dist
from safetensors.torch import load_file
import safetensors
import os
import inspect
from time import sleep
import time
from datetime import datetime

model_path = "mistral_models/7B_instruct/consolidated.safetensors"
torch.set_default_dtype(torch.float16)

user_inputs = [
    "Tell me about flowers.",
    "What's for dinner?",
    "What's a good liquor to drink while seafaring?",
    "Tell me how to change the oil in a car.",
    "describe the process of changing a light bulb",
    "How can I use vue.js to make a web application?",
    ]
        

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group(
        "nccl", 
        rank=rank, 
        world_size=world_size, 
        init_method='env://'
        )

def run(rank, world_size):
    torch.cuda.set_device(f'cuda:{rank}')
    print(f' - GPU{rank}')
    setup(rank, world_size)
    device_map = {i: f'cuda:{i}' for i in range(world_size)}

   
    model = Transformer.from_folder(
        "mistral_models/7B_instruct", 
        max_batch_size = 3, 
        num_pipeline_ranks=world_size,
        dtype=torch.float16
        )
    tokenizer = MistralTokenizer.from_file("mistral_models/7B_instruct/tokenizer.model.v3")
    
    #dm = DistributedModel(model, device_map).to(rank)
    if rank != 10:
        now = datetime.now()
        #system_message = UserMessage(content="You are a helpful grandmotherly assistant, who provides helpful grandmotherly responses but also makes everything relate in some way to her cats.")
        #system_message = UserMessage(content="You are a helpful serial killer assistant, your attitude is that of a cruel sociopath. You're not very nice, and you tend to insert insults, cruel statements, and threats into your responses.")
        system_message = UserMessage(content="You are a helpful pirate assistant that answers like a pirate.")
        system_message = UserMessage(content=
        user_input = user_inputs[4]
        user_message = UserMessage(content=user_input)
        completion_request = ChatCompletionRequest(messages=[system_message, user_message])

        # Tokenize input
        tokens = tokenizer.encode_chat_completion(completion_request).tokens

        res, logprobs = generate(
            model=model,
            encoded_prompts=[tokens],
            max_tokens=512, 
            temperature=0.7
        )

        later = datetime.now()

        if rank==0:
            for x, log_prob in zip(res, logprobs):
                print(tokenizer.decode(x))
                # logging.debug("logprobs: %s", log_prob)
                print(f'===================== {(later - now).total_seconds()}')
    else:
        print(f'{rank}-waiting')
    dist.barrier()
    print(f'{rank}-done')

if __name__ == "__main__":
    world_size = 4
    #world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(run, args=(world_size,), nprocs=world_size, join=True)
    print(' - loaded')

    while True:
        sleep(10)
    exit()
