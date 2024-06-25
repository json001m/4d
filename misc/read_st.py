import json
import struct
from pprint import pp
from mistral_inference.model import Transformer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_inference.generate import generate
from accelerate import Accelerator
import torch
from torch import nn, distributed as dist
from safetensors.torch import load_file
import os
import inspect

os.environ['TORCH_CPP_LOG_LEVEL'] = 'INFO'

model_path = "mistral_models/7B_instruct/consolidated.safetensors"
num_layers = 32
num_gpus = 6

# Don't need this. Just playing around to read the header and see what's in there
with open(model_path, "rb") as f:
    length_of_header = struct.unpack('<Q', f.read(8))[0]
    header_data = f.read(length_of_header)
    header = json.loads(header_data)

#pp(header)  # should be a dict that contains what you need

# load the tensors into memory
print('loading model...')
model_path = "mistral_models/7B_instruct/consolidated.safetensors"
tensors = load_file(model_path)
print(f' - loaded model type: {tensors[0].type()}')

# chunker for distribution
def chunk_tensor(tensor, num_chunks):
    return torch.chunk(tensor, num_chunks)

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

# Example usage
def chunk_tensor(tensor, num_chunks):
    return torch.chunk(tensor, num_chunks)

# Example chunked_tensors dict (this should be created by loading and chunking your model parameters)
# chunked_tensors = {
#     'layers.21.attention.wk.weight': [torch.randn(100, 200) for _ in range(6)],
#     'layers.21.attention.wk.bias': [torch.randn(100) for _ in range(6)],
#     # Add other parameter chunks here...
# }

device_map = {i: f'cuda:{i}' for i in range(num_gpus)}

# # mistral 7B has 32 layers
# for layer_num in range(32):
#     print(f'creating layer {layer_num}...')
#     distributed_layer = create_distributed_layer(layer_num, chunked_tensors, device_map)

class DistributedTransformerLayer(nn.Module):
    def __init__(self, distributed_params):
        super(DistributedTransformerLayer, self).__init__()
        
        self.attention_weights = nn.ParameterDict()
        self.ffn_weights = nn.ParameterDict()
        
        for name, params in distributed_params.items():
            for i, (device, param) in enumerate(params):
                if 'attention' in name:
                    #print(f'{type.mro(nn.Parameter(param))}')
                    print(f'{param.type()}')
                    # inspect.getclass(nn.Parameter(param).type())
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

# Reconstructing the distributed model with multiple layers
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


num_layers = 32  
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