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

model_path = "mistral_models/7B_instruct/consolidated.safetensors"
torch.set_default_dtype(torch.float16)

def create_distributed_layer(layer_num, chunked_tensors, device_map):
    # Define the parameter names expected in each layer of the Mistral model
    param_names = [
        'attention.wk.weight', 'attention.wq.weight', 'attention.wv.weight', 'attention.wo.weight',
        'attention.wk.bias', 'attention.wq.bias', 'attention.wv.bias', 'attention.wo.bias',
        'feed_forward.fc1.weight', 'feed_forward.fc1.bias',
        'feed_forward.fc2.weight', 'feed_forward.fc2.bias'
    ]

    layer_params = {}

    for param_name in param_names:
        full_param_name = f'layers.{layer_num}.{param_name}'
        if full_param_name in chunked_tensors:
            param_chunks = chunked_tensors[full_param_name]
            distributed_params = [(device_map[i], nn.Parameter(chunk.to(device_map[i]))) for i, chunk in enumerate(param_chunks)]
            layer_params[param_name.replace('.', '_')] = distributed_params
    
    return layer_params

class DistributedTransformerLayer(nn.Module):
    def __init__(self, distributed_params):
        super(DistributedTransformerLayer, self).__init__()
        
        self.attention_weights = nn.ParameterDict()
        self.ffn_weights = nn.ParameterDict()
        
        for name, params in distributed_params.items():
            for i, (device, param) in enumerate(params):
                if 'attention' in name:
                    self.attention_weights[f'{name}_{i}'] = param
                elif 'feed_forward' in name:
                    self.ffn_weights[f'{name}_{i}'] = param

    def forward(self, x):
        # Example forward pass: applying attention and feed-forward networks
        # This is a simplified example; actual implementation depends on model specifics
        for name, param in self.attention_weights.items():
            device = param.device
            x = x.to(device)
            x = torch.matmul(x, param)
        
        for name, param in self.ffn_weights.items():
            device = param.device
            x = x.to(device)
            x = torch.matmul(x, param)
        
        return x

class DistributedModel(nn.Module):
    def __init__(self, num_layers, chunked_tensors, device_map):
        super(DistributedModel, self).__init__()
        self.layers = nn.ModuleList()
        for layer_num in range(num_layers):
            print(f' - layer {layer_num}')
            distributed_params = create_distributed_layer(layer_num, chunked_tensors, device_map)
            layer = DistributedTransformerLayer(distributed_params)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Don't need this. Just playing around to read the header and see what's in there
with open(model_path, "rb") as f:
    length_of_header = struct.unpack('<Q', f.read(8))[0]
    header_data = f.read(length_of_header)
    header = json.loads(header_data)
#pp(header)  

# load the model from disk and prepare it for use
print('loading model...')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

print('Setting up model across 6 GPUs...')
def run(rank, world_size):
    torch.cuda.set_device(f'cuda:{rank}')
    print(f' - GPU{rank}')
    setup(rank, 6)
    model = Transformer.from_folder(
        "mistral_models/7B_instruct", 
        max_batch_size = 3, 
        num_pipeline_ranks=world_size,
        dtype=torch.float16
        )


    tokenizer = MistralTokenizer.from_file("mistral_models/7B_instruct/tokenizer.model.v3")
    #tokenizer: Tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer   

    system_message = UserMessage(content="You are a helpful pirate assistant that answers like a pirate.")
    user_input = "Tell me a joke."
    user_message = UserMessage(content=user_input)
    completion_request = ChatCompletionRequest(messages=[system_message, user_message])

    # Tokenize input
    tokens = tokenizer.encode_chat_completion(completion_request).tokens


    res, logprobs = generate(
        model=model,
        encoded_prompts=[tokens],
        max_tokens=64, 
        temperature=0.7
    )

    for x, log_prob in zip(res, logprobs):
        print(tokenizer.decode(x))
        # logging.debug("logprobs: %s", log_prob)
        print("=====================")

if __name__ == "__main__":
    os.environ['TORCH_LOG_LEVEL'] = 'INFO'

    
    num_layers = 32
    num_gpus = 6
    max_batch_size = 3
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(run, args=(world_size,), nprocs=world_size, join=True)
    print(' - loaded')

    while True:
        sleep(1)
    exit()
### COME BACK TO THIS
# model_path = "mistral_models/7B_instruct/consolidated.safetensors"

# with open("mistral_models/7B_instruct/params.json", "r") as f:
#     model_args = ModelArgs.from_dict(json.load(f))
# model_args.max_batch_size = max_batch_size

# model = Transformer(model_args)

# safetensors_model_file = "mistral_models/7B_instruct/consolidated.safetensors"

# loaded_tensors = load_file(str(safetensors_model_file))
# model.load_state_dict(loaded_tensors, assign=True, strict=True)

# #mistral_model = model.to(device=device, dtype=dtype)
# print(f' - loaded model: mistral-7B-instruct')

# # chunker for distribution
# def chunk_tensor(tensor, num_chunks):
#     return torch.chunk(tensor, num_chunks)

# print('chunking tensors...')
# chunked_tensors = {name: chunk_tensor(tensor, num_gpus) for name, tensor in loaded_tensors.items()}
# print(' - done')

'''

device_map = {i: f'cuda:{i}' for i in range(num_gpus)}

# # mistral 7B has 32 layers
# for layer_num in range(32):
#     print(f'creating layer {layer_num}...')
#     distributed_layer = create_distributed_layer(layer_num, chunked_tensors, device_map)



# Reconstructing the distributed model with multiple layers


# Example usage
num_layers = 32  # Adjust based on your model
distributed_model = DistributedModel(num_layers, chunked_tensors, device_map)
#synchronize()

tokenizer = MistralTokenizer.from_file("mistral_models/7B_instruct/tokenizer.model.v3")

# Tokenize the input text
system_message = UserMessage(content="You are a helpful pirate assistant that answers like a pirate.")

user_input = "tell me a story."

user_message = UserMessage(content=user_input)
completion_request = ChatCompletionRequest(messages=[system_message, user_message])

input_ids = tokenizer.encode_chat_completion(completion_request).tokens
#print(f'{input_ids.type()}')
# pp(input_ids)
# pp(distributed_model.modules)
# print(f'{inspect.getmembers(distributed_model)}')
# output_ids, _ = generate([input_ids], distributed_model, max_tokens=64, temperature=0.7)

#input_ids = input_ids.to('cuda:0')
output_ids = torch.tensor(input_ids, dtype=torch.float16).unsqueeze(0).to('cuda:0')
# Generate text
max_length = 50
#output_ids = input_ids

for _ in range(max_length):
    with torch.no_grad():
        output = distributed_model(output_ids)
    
    next_token_logits = output[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    output_ids = torch.cat([output_ids, next_token_id], dim=-1)

    if next_token_id.item() == tokenizer.eos_token_id:
        break

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)
'''
