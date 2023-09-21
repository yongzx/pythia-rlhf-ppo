import transformers
import torch

model = transformers.GPTNeoXForCausalLM.from_pretrained("/mnt/ssd-1/pythia-rlhf/checkpoints/sft_hh/pythia-70m/checkpoint_14000")
tokenizer = transformers.AutoTokenizer.from_pretrained("/mnt/ssd-1/pythia-rlhf/checkpoints/sft_hh/pythia-70m/checkpoint_14000")
inputs = tokenizer(["I am here."], return_tensors="pt")
print(inputs)
outputs = model.generate(**inputs, max_new_tokens=10)
# print(outputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])