import json
import struct
from pprint import pp
from mistral_inference.model import Transformer, ModelArgs
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.messages import SystemMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_inference.generate import generate
from accelerate import Accelerator
import torch
import torch.quantization
from torch import nn, distributed as dist
from safetensors.torch import load_file
import safetensors
import os
import inspect
from time import sleep
import time
from datetime import datetime
import random
import numpy as np

model_path = "../model_server/models/mistral/7B_instruct/consolidated.safetensors"
torch.set_default_dtype(torch.float16)

user_inputs = [
    "a detailed explaination of dark energy?",
    "a detailed explaination of general relativity?",
    "a detailed explaination of special relativity?",
    "a detailed explaination of the process of building a house?",
    "a detailed explaination of how to make creme fraiche?",
    "can you explain in detail how to change a tire?"
    ]
        

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    
    seed = 44
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init_process_group(
        "nccl", 
        rank=rank, 
        world_size=world_size, 
        init_method='env://'
        )

def mistral_7b_generate(prompt, rank):
    now = datetime.now()

    model = Transformer.from_folder(
        "../model_server/models/mistral/7B_instruct", 
        max_batch_size = 3, 
        num_pipeline_ranks=6,
        dtype=torch.float16
        )

    # quantized_model = torch.quantization.quantize_dynamic(
    #     model,  # the original model
    #     {torch.nn.Linear},  # layers to quantize
    #     dtype=torch.qint8  # quantization dtype
    # )

    tokenizer = MistralTokenizer.from_file("../model_server/models/mistral/7B_instruct/tokenizer.model.v3")
    system_message = SystemMessage(content="You are a helpful assistant. You will always provide a response that is just one word, either 'yes' or 'no'. You will stop responding after this word, and provide no further explaination.")
    #system_message = SystemMessage(content="You are a helpful assistant. Inputs will be provided to you in the following format: 'Subject !! statement'. You will assess the statement determine if it represents a positive, negative, or neutral sentiment about the subject, and always provide a response that is just one word, either 'positive', 'negative', or 'neutral'. You will stop responding after this word, and provide no further explaination.")
    
    
    user_input = prompt
    user_message = UserMessage(content=user_input)
    completion_request = ChatCompletionRequest(messages=[system_message, user_message])

    # Tokenize input
    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    res, logprobs = generate(
        model=model,
        encoded_prompts=[tokens],
        max_tokens=2048, 
        temperature=0.7,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id
    )
    
    later = datetime.now()

    if rank==0:
        for x, log_prob in zip(res, logprobs):
            sleep(rank*.5)
            print(tokenizer.decode(x))
            # logging.debug("logprobs: %s", log_prob)
            print(f'===================== {(later - now).total_seconds()}')

    return res, logprobs

def run(rank, world_size, prompt):
    torch.cuda.set_device(f'cuda:{rank}')
    print(f' - GPU{rank}')
    setup(rank, world_size)
    device_map = {i: f'cuda:{i}' for i in range(world_size)}

    res, logprobs = mistral_7b_generate(user_inputs[prompt], rank)

    print(f'{rank}-done')
    dist.barrier()
    

if __name__ == "__main__":
    #for x in range(6):
    #world_size = torch.cuda.device_count()
    world_size = 3
    torch.multiprocessing.spawn(run, args=(world_size, 5), nprocs=world_size, join=True)
    print(' - fin')

    # now = datetime.now()
    # mistral_7b_generate(user_inputs[3])
    # later = datetime.now()
    # for x, log_prob in zip(res, logprobs):
    #     print(tokenizer.decode(x))
    #     # logging.debug("logprobs: %s", log_prob)
    #     print(f'===================== {(later - now).total_seconds()}')

    while True:
        sleep(.1)
    exit()
