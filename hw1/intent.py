import pandas as pd
import numpy as np
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models


def main(args):

    # load test and pickle file 
    test = pd.read_json(args.test_file)
    with open("intent_tokenizer.pkl", "rb") as f:
      tokenizer = pickle.load(f)
    with open("label_intent.pkl", "rb") as f:
      label_intent = pickle.load(f)
    
    # preprocessing
    test_seq = tokenizer.texts_to_sequences(test.text)

    # padding 
    max_len = 28
    for s in test_seq:
        while len(s) < max_len:
            s.append(0)
    X_test = np.array(test_seq)
    
    # load best model
    model = models.load_model(args.ckpt_path)
    pre = model.predict(X_test)
    predict = [np.argmax(a) for a in pre]
    pre_intent = [label_intent["intent"][a] for a in predict]

    df = pd.DataFrame()
    df["id"] = test.id
    df["intent"] = pre_intent
    df.to_csv(args.pred_file, index=False)   

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
        default = "./Intent.h5"
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)