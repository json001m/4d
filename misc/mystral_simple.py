import torch
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference import generate

# Initialize the tokenizer
tokenizer = MistralTokenizer.from_pretrained('mistral_7b_instruct')

# Load the model (replace 'path_to_mistral_7b_instruct_model' with the actual model path)
model = torch.load('path_to_mistral_7b_instruct_model')

# Check the number of available GPUs
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    raise RuntimeError("No GPUs available")

# Split the model across multiple GPUs
model = torch.nn.DataParallel(model)

# Move the model to GPU
model.to('cuda')

def inference(input_text):
    # Tokenize the input text
    inputs = tokenizer.encode(input_text, return_tensors='pt')

    # Move inputs to GPU
    inputs = inputs.to('cuda')

    # Generate output
    with torch.no_grad():
        outputs = generate(model, inputs)

    # Decode the output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text

# Example usage
input_text = "Your input text here"
output_text = inference(input_text)
print(output_text)