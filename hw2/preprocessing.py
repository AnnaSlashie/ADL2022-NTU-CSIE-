import os
import sys
import json
import datasets
from dataclasses import dataclass, field
from datasets import load_dataset
from pathlib import Path
from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
  parser = ArgumentParser()
  parser.add_argument('--test_file', type=Path, help='Path to the test file.',default='./test.json', required=True)
  parser.add_argument('--context_file', type=Path, help='Path to the context file.',default='./context.json', required=True)
  args = parser.parse_args()    
  return args

args = parse_args()

#loading context_file
with open(args.context_file,"r") as f :
  context = json.load(f)

def encoder(texts):
    num_examples = len(texts['id'])
    return{
        "video-id":texts['id'],
        "fold-ind":texts['id'],
        "sent1":[question for question in texts['question']],
        "sent2":[""]*num_examples,
        **{f"ending{i}":[ context[idx] for idx in list(para_list)] for i, para_list in enumerate(zip(*texts["paragraphs"]))},
        "label": [0] * num_examples
    }

#loading test_file  
test_data = {'test': str(args.test_file)}
test_dataset = load_dataset('json', data_files = test_data, field = slice(None))
test_dataset = test_dataset["test"]

#save file after preprocessing
savefile = "mc_test"
test_dataset = test_dataset.map(encoder, batched=True, batch_size=512, remove_columns=test_dataset.column_names)
test_dataset.to_json(f"{savefile}.json")

