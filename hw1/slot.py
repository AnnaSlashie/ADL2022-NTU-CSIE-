import pandas as pd
import numpy as np
import json
import pickle
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models


def main(args):
    
    # load test and pickle file 
    test = pd.read_json(args.test_file)
    test["tokens_lens"] = test.tokens.apply(lambda x: len(x))
    
    with open("slot_tokenizer.pkl", "rb") as f:
      tokenizer = pickle.load(f)
    with open("label_tag.pkl", "rb") as f:
      label_tag = pickle.load(f)

    #conduct padding
    max_len = 35
    def padding(token):
      while len(token)<max_len: 
        token.append("padding")
    test.tokens.apply(padding)

    # preprocessing
    test.sequence = test.tokens.apply(lambda x:" ".join(x))
    test_seq = tokenizer.texts_to_sequences(test.tokens)
    for s in test_seq:
      while len(s)< max_len:
        s.append(1)
    X_test = np.array(test_seq)
    test_dataset = tf.data.Dataset.from_tensor_slices(
    tf.cast(X_test, tf.int64),
    )
    test_dataset = test_dataset.batch(256)

    # load model ckpt
    model = models.load_model(args.ckpt_path)

    def evaluate_sentence(inputs):
      inference_batch_size = inputs.shape[0]
      outputs = model(inputs, training=False)
      outputs = tf.argmax(outputs, axis=-1)
      return outputs

    preds = []
    for inputs in tqdm(test_dataset, total=len(test_dataset)):
      outputs = evaluate_sentence(inputs)
      preds.append(outputs)
    
    preds = np.vstack(preds)
    preds = [p for p in preds]
    test["preds"] = preds

    #reset to original length
    for i in range(len(test)):
      test["preds"][i] = test["preds"][i][:test.tokens_lens[i]] 
    def preds2tags(pred):
      tag = []
      for p in pred:
        t = label_tag[p]
        if t == "padding" :
          t = "O"
        tag.append(t)
      return tag

    #Submission
    df = pd.DataFrame()
    df["id"] = test.id
    df["tags"] = test.preds.apply(preds2tags)
    df["tags"] = df.tags.apply(lambda x: " ".join(x))

    df.to_csv(args.pred_file ,index=False)
  
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )

    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True,
        default = "./Slot.h5"
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
