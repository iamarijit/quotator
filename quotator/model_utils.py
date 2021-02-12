import torch
from torch import nn
import numpy as np
from quotator.model_params import *

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.vocab_size = len(word_to_int)
        self.lstm_size = LSTM_SIZE
        self.embedding_dim = EMB_SIZE
        self.num_layers = NUM_LAYERS

        self.emb = nn.Embedding(
            num_embeddings=self.vocab_size, 
            embedding_dim=self.embedding_dim
            )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers
            )
        self.fc = nn.Linear(self.lstm_size, self.vocab_size)

    def forward(self, x, prev_state):
        emb = self.emb(x)
        output, state = self.lstm(emb, prev_state)
        y = self.fc(output)
        return y, state

    def init_state(self, seq_length):
        return (
            torch.zeros(self.num_layers, seq_length, self.lstm_size).to(DEVICE),
            torch.zeros(self.num_layers, seq_length, self.lstm_size).to(DEVICE)
        )

def generate(max_words=MAX_WORDS):
    model = return_model()
    model.eval()
    h, c = model.init_state(seq_length=1)

    x = torch.from_numpy(np.array([word_to_int[START_TOKEN]])).to(DEVICE).long()
    x = x.unsqueeze(0)

    words = []
    for w in range(max_words):
        y, (h, c) = model(x, (h, c))        
        y = y[0][-1]
        
        p = nn.functional.softmax(y, dim=0).cpu().detach().numpy()
        word_index = np.random.choice(len(y), p=p)

        while word_index == word_to_int[SPECIAL_TOKEN]:
            word_index = np.random.choice(len(y), p=p)        

        if int_to_word[word_index] == END_TOKEN or int_to_word[word_index] == PAD_TOKEN:
            break

        x = torch.from_numpy(np.array([word_index])).to(DEVICE).long()
        x = x.unsqueeze(0)

        words.append(int_to_word[word_index])

    quote = " ".join(words).capitalize()
    return quote

def return_model():
    model = Model()
    model.load_state_dict(torch.load(MODEL_DIR, map_location=DEVICE))
    return model