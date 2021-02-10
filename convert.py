import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print(":: Loading the model")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

with open("text.en", "r") as f:
  input_encoder = tokenizer(f.read(), return_tensors="pt")

print(":: Converting to ONNX runtime")
torch.onnx.export(model, input_encoder["input_ids"], "gpt2.onnx")
