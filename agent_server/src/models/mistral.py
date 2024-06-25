from mistral_inference.model import Transformer, ModelArgs
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.messages import SystemMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_inference.generate import generate

import torch
import torch.quantization
from torch import nn, distributed as dist
from safetensors.torch import load_file

import multiprocessing as mp
import os
import sys
from time import sleep
from datetime import datetime, timedelta
import random
import numpy as np
from pprint import pp
import logging

dirname = os.path.dirname(__file__)
model_path = dirname + "/../../models/mistral/7B_instruct"
torch.set_default_dtype(torch.float16)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    
    seed = 42
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

def mistral_7b_generate(prompt, rank, world_size, system_message="You are a helpful assistant.", temperature=0.7):
    now = datetime.now()

    model = Transformer.from_folder(
        model_path, 
        max_batch_size = 3, 
        num_pipeline_ranks = world_size,
        dtype=torch.float16
        )

    tokenizer = MistralTokenizer.from_file(f'{model_path}/tokenizer.model.v3')

    # get the full model prompt put together
    _system_message = SystemMessage(content=system_message)
    user_message = UserMessage(content=prompt)
    completion_request = ChatCompletionRequest(messages=[_system_message, user_message])

    # Tokenize input
    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    # execute the model
    res, logprobs = generate(
        model=model,
        encoded_prompts=[tokens],
        max_tokens=2048, 
        temperature=temperature,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id
    )
    text_response = ""
    later = datetime.now()
    print('hello')
    if rank==0:
        for x, log_prob in zip(res, logprobs):
            text_response = tokenizer.decode(x)
            logging.info(f' - Request "{prompt}" responded in {(later - now).total_seconds()}')
            print(f' - Request "{prompt}" responded in {(later - now).total_seconds()}')
            logging.debug(f' - Response: \n\n {text_response}')
            print(f' - Response: \n\n {text_response}')

    return text_response, res, logprobs

def run(rank, prompt, world_size, system_message, temperature):
    torch.cuda.set_device(f'cuda:{rank}')
    logging.debug(f' - spinning up GPU{rank}.')
    setup(rank, world_size)

    client_store = dist.TCPStore("localhost", 12344, 7, False)
    tr, res, logprobs = mistral_7b_generate(prompt, rank, world_size, system_message=system_message, temperature=temperature)
    if rank==0:
        client_store.set("complete", "true")
        client_store.set("result", tr)
    
    logging.debug(f' - GPU{rank} complete.')
    dist.barrier()
    
# TODO: This really needs to return an ack immediately, and then send an async message when the model completes.
#       It will work for now but long term this is inefficient
def query(prompt, system_message=None, temperature=0.7, context=None):
    torch.set_default_dtype(torch.float16)
    logging.info(f'Received query request: {prompt}')

    # TODO: Make this something that can be programmatically determined... Should only need 3 GPUs to run mistral,
    #       so should be able to run two copies if we're globally keeping track of GPU utilization
    world_size = torch.cuda.device_count()

     # Set up IPC 
    master_store = dist.TCPStore("localhost", 12344, 7, True, timedelta(seconds=30), False, False) 

    torch.multiprocessing.spawn(run, args=(prompt, world_size, system_message, temperature), nprocs=world_size, join=True)
    
    # TODO: replace this with a different form of comms / polling. 
    query_satisfied = False
    while not query_satisfied:
        sleep(.05)
        query_satisfied = (master_store.get("complete") == b'true')
    result = master_store.get("result")

    # TODO: what's the best response format????
    return result.decode("utf-8")

if __name__ == "__main__":
    # Set up logging formatting, set to use STDOUT and also to use DEBUG loglevel
    logging.basicConfig(level=logging.DEBUG)
    # root = logging.getLogger()
    # root.setLevel(logging.DEBUG)
    # handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    # root.addHandler(handler)

    world_size = 6 #torch.cuda.device_count()

    # Yes/No queries
    #system_message = "You are a helpful assistant. You will always provide a response that is just one word, either 'yes' or 'no'. You will stop responding after this word, and provide no further explaination."
    # Sentiment Analysis
    #system_message = "You are a helpful assistant. Inputs will be provided to you in the following format: 'Subject !! statement'. You will assess the statement determine if it represents a positive, negative, or neutral sentiment about the subject, and always provide a response that is just one word, either 'positive', 'negative', or 'neutral'. You will stop responding after this word, and provide no further explaination."
    system_message = "You are a helpful assistant. The user will provide an exact quote from a movie script. Your response will be a list of only the titles of the movies that contain that quote, separated by commas. You will provide no further output under any circumstances. The exact quote you are to identify will be the next input."
    
    prompt = "'Frankly, my dear, I don't give a damn.'" # YES "Gone With the Wind"
    
    # Set up IPC 
    master_store = dist.TCPStore("localhost", 12344, 7, True, timedelta(seconds=30), False, False) 

    # Kick this bitch off
    torch.multiprocessing.spawn(run, args=(prompt, world_size, system_message, 0.7), nprocs=world_size, join=True)
    print('done')
    query_satisfied = False
    while not query_satisfied:
        sleep(1)
        query_satisfied = (master_store.get("complete") == b'true')
    result = master_store.get("result")
    print(f'********** RESULT:\n{result}\n************')
    exit()
