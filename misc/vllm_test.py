from vllm import LLM, SamplingParams
import torch

torch.set_default_dtype(torch.float16)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

model_dir = 'mistral_models/7B_instruct/'

llm = LLM(model=model_dir,  dtype=torch.float16, tensor_parallel_size=4) #gpu_memory_utilization=0.5,

utputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
