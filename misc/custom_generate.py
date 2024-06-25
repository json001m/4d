
from mistral_inference.model import Transformer, ModelArgs
from mistral_inference.cache import BufferCache
from mistral_inference.model import Transformer
from mistral_inference.generate import sample
# from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

from accelerate import Accelerator
import torch
from torch import nn, distributed as dist
from safetensors.torch import load_file
import safetensors

import json
import struct
from pprint import pp
import os
import inspect
from time import sleep
import time
from datetime import datetime
import random
import numpy as np
from typing import List, Optional, Tuple


model_path = "mistral_models/7B_instruct/consolidated.safetensors"
torch.set_default_dtype(torch.float16)

user_inputs = [
    "Tell me about flowers.",
    "What's for dinner?"
    ]

def generate(
    encoded_prompts: List[List[int]],
    model: Transformer,
    *,
    max_tokens: int,
    temperature: float,
    chunk_size: Optional[int] = None,
    eos_id: Optional[int] = None
) -> Tuple[List[List[int]], List[List[float]]]:
    model = model.eval()
    B, V = len(encoded_prompts), model.args.vocab_size

    seqlens = [len(x) for x in encoded_prompts]

    # Cache
    cache_window = max(seqlens) + max_tokens
    cache = BufferCache(
        model.n_local_layers,
        model.args.max_batch_size,
        cache_window,
        model.args.n_kv_heads,
        model.args.head_dim,
    )

    cache.to(device=model.device, dtype=model.dtype)
    cache.reset()

    # Bookkeeping
    logprobs: List[List[float]] = [[] for _ in range(B)]
    last_token_prelogits = None

    # One chunk if size not specified
    max_prompt_len = max(seqlens)
    if chunk_size is None:
        chunk_size = max_prompt_len

    # Encode prompt by chunks
    for s in range(0, max_prompt_len, chunk_size):
        prompt_chunks = [p[s : s + chunk_size] for p in encoded_prompts]
        assert all(len(p) > 0 for p in prompt_chunks)
        prelogits = model.forward(
            torch.tensor(sum(prompt_chunks, []), device=model.device, dtype=torch.long),
            seqlens=[len(p) for p in prompt_chunks],
            cache=cache,
        )
        logits = torch.log_softmax(prelogits, dim=-1)

        if last_token_prelogits is not None:
            # Pass > 1
            last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
            for i_seq in range(B):
                logprobs[i_seq].append(
                    last_token_logits[i_seq, prompt_chunks[i_seq][0]].item()
                )

        offset = 0
        for i_seq, sequence in enumerate(prompt_chunks):
            logprobs[i_seq].extend(
                [
                    logits[offset + i, sequence[i + 1]].item()
                    for i in range(len(sequence) - 1)
                ]
            )
            offset += len(sequence)

        last_token_prelogits = prelogits.index_select(
            0,
            torch.tensor(
                [len(p) for p in prompt_chunks], device=prelogits.device
            ).cumsum(dim=0)
            - 1,
        )
        assert last_token_prelogits.shape == (B, V)

    # decode
    generated_tensors = []
    is_finished = torch.tensor([False for _ in range(B)])

    assert last_token_prelogits is not None
    for _ in range(max_tokens):
        next_token = sample(last_token_prelogits, temperature=temperature, top_p=0.8)

        if eos_id is not None:
            is_finished = is_finished ^ (next_token == eos_id).cpu()

        if is_finished.all():
            break

        last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
        for i in range(B):
            logprobs[i].append(last_token_logits[i, next_token[i]].item())

        generated_tensors.append(next_token[:, None])
        last_token_prelogits = model.forward(next_token, seqlens=[1] * B, cache=cache)
        assert last_token_prelogits.shape == (B, V)

    generated_tokens: List[List[int]]
    if generated_tensors:
        generated_tokens = torch.cat(generated_tensors, 1).tolist()
    else:
        generated_tokens = []

    return generated_tokens, logprobs   

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    
    seed = 43
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
        "mistral_models/7B_instruct", 
        max_batch_size = 3, 
        num_pipeline_ranks=6,
        dtype=torch.float16
        )
    tokenizer = MistralTokenizer.from_file("mistral_models/7B_instruct/tokenizer.model.v3")
       
    #system_message = UserMessage(content="You are a helpful grandmotherly assistant, who provides helpful grandmotherly responses but also makes everything relate in some way to her cats.")
    #system_message = UserMessage(content="You are a helpful serial killer assistant, your attitude is that of a cruel sociopath. You're not very nice, and you tend to insert insults, cruel statements, and threats into your responses.")
    system_message = UserMessage(content="You are a helpful pirate assistant that answers like a pirate. You have a peg leg and a hook hand, and leverage these in your responses, usually.")
    
    user_input = prompt
    user_message = UserMessage(content=user_input)
    completion_request = ChatCompletionRequest(messages=[system_message, user_message])

    # Tokenize input
    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    res, logprobs = generate(
        model=model,
        encoded_prompts=[tokens],
        max_tokens=512, 
        temperature=0.7,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id
    )
    
    later = datetime.now()

    if rank==0:
        for x, log_prob in zip(res, logprobs):
            print(tokenizer.decode(x))
            # logging.debug("logprobs: %s", log_prob)
            print(f'===================== {(later - now).total_seconds()}')

    return res, logprobs

def run(rank, world_size):
    torch.cuda.set_device(f'cuda:{rank}')
    print(f' - GPU{rank}')
    setup(rank, world_size)
    device_map = {i: f'cuda:{i}' for i in range(world_size)}

    res, logprobs = mistral_7b_generate(user_inputs[4], rank)

    print(f'{rank}-done')
    dist.barrier()
    
if __name__ == "__main__":
    #world_size = 6
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(run, args=(world_size,), nprocs=world_size, join=True)
    print(' - loaded')

    # now = datetime.now()
    # mistral_7b_generate(user_inputs[3])
    # later = datetime.now()
    # for x, log_prob in zip(res, logprobs):
    #     print(tokenizer.decode(x))
    #     # logging.debug("logprobs: %s", log_prob)
    #     print(f'===================== {(later - now).total_seconds()}')

    while True:
        sleep(10)
    exit()
