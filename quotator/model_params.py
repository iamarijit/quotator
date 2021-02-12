import pandas as pd
import torch
import os

START_TOKEN = "<str>"
END_TOKEN = "<end>"
PAD_TOKEN = "<pad>"
SPECIAL_TOKEN = "<spc>"

MAX_QUOTE_LEN = 8
EMB_SIZE = 256
NUM_LAYERS = 1
LSTM_SIZE = 256
MAX_WORDS = 6

VOCAB_DIR = os.path.join('quotator', 'vocab.csv')
MODEL_DIR = os.path.join('quotator', 'model.pt')

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda") if CUDA else torch.device("cpu")

DIC = pd.read_csv(VOCAB_DIR)

words = list(DIC.iloc[:, 1])
ids = list(DIC.iloc[:, 0])

word_to_int = dict(zip(words, ids))
int_to_word = dict(zip(ids, words))