#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import os
import time
import numpy as np
from argparse import ArgumentParser

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from openvino.inference_engine import IECore

def generate_greedy_pytorch(tokens, model, n):
  complete_seq = tokens.permute((1, 0)).tolist()
  for _ in range(n):
    out = model(tokens)
    next_tokens = torch.argmax(out.logits[:, -1], dim = -1).unsqueeze(1)
    tokens = torch.cat([tokens, next_tokens], dim=-1)
    tokens = tokens[:, 1:]
    complete_seq.extend(next_tokens.tolist())
  return np.array(complete_seq).T.tolist()


def generate_greedy_openvino(tokens, exec_net, n, logits_dict_key = "2859"):
  complete_seq = tokens.T.tolist()
  for _ in range(n):
    out = exec_net.infer(inputs={"0": inputs})[logits_dict_key]
    next_tokens = np.argmax(out[:, -1], axis=-1).reshape(-1, 1)
    tokens = np.hstack((tokens, next_tokens))
    tokens = tokens[:, 1:]
    complete_seq.extend(next_tokens.tolist())
  return np.array(complete_seq).T.tolist()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("--model", help="Path to an .xml file with a trained model.", default = "./gpt2.xml", type=str)
  parser.add_argument("--g", help="if set model will also test generation", action = "store_true", default = False)
  args = parser.parse_args()

  print("-"*70)
  print("Loading Pytorch model")
  tokenizer = AutoTokenizer.from_pretrained("gpt2")
  model = AutoModelForCausalLM.from_pretrained("gpt2")
  with open("text.en", "r") as f:
    text = f.read()
  input_encoder = tokenizer([text + tokenizer.eos_token for _ in range(1)], return_tensors="pt")
  print("Text shape:", input_encoder["input_ids"].size())

  st = time.time()
  model(input_encoder[ "input_ids"])
  print(f":: Pytorch inference in {time.time() - st:.5f}s")
  if args.g:
    print("-"*70)
    print("Testing generation")
    st = time.time()
    out = generate_greedy_pytorch(input_encoder["input_ids"], model, n = 40)
    out = tokenizer.decode(out[0])
    print(f":: Pytorch generation took (40 steps): {time.time() - st:.3f}s")
  del model

  print("-"*70)
  model_xml = args.model
  model_bin = os.path.splitext(model_xml)[0] + ".bin"

  # Plugin initialization for specified device and load extensions library if specified.
  print("Creating Inference Engine...")
  ie = IECore()

  # Read IR
  print("Loading network")
  net = ie.read_network(args.model, os.path.splitext(args.model)[0] + ".bin")

  print("Loading IR to the plugin...")
  exec_net = ie.load_network(network=net, device_name="CPU", num_requests=2)
  print(f"exec_net: {exec_net}")

  # this is a bit tricky. So the input to the model is the input from ONNX graph
  # IECore makes a networkX graph of the "computation graph" and when we run .infer
  # it passes it through. If you are unsure of what to pass you can always check the
  # <model>.xml file. In case of pytorch models the value "input.1" is the usual
  # suspect. Happy Hunting!
  inputs = input_encoder["input_ids"].tolist()
  st = time.time()
  out = exec_net.infer(inputs={"0": inputs}, )
  
  # now this out is a dictionary and has a lot of outputs so you will need to manually
  # determine which is the output that you want by checking the correct shape
  # for k in list(out.keys()):
  #   print(k, "-->", out[k].shape)

  print(f":: OpenVino inference in {time.time() - st:.5f}s")

  if args.g:
    print("-"*70)
    print("Testing generation")
    st = time.time()
    out = generate_greedy_openvino(input_encoder["input_ids"].numpy(), exec_net, n=40)
    out = tokenizer.decode(out[0])
    print(f":: OpenVino generation took (40 steps): {time.time() - st:.3f}s")

  print("-"*70)
