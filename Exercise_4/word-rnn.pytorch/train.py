#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import math
import matplotlib.pyplot as plt

from tqdm import tqdm

from helpers import *
from model import *
from generate import *
from vocab import Vocabulary

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=4000)
argparser.add_argument('--print_every', type=int, default=400)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=20)
argparser.add_argument('--batch_size', type=int, default=16)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--cuda', action='store_true')
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

voc = Vocabulary('voc')
file, file_len = read_file(args.filename)                                   # Reads file as giant string

for w in file:
    voc.add_word(w)
print(voc.num_words, "words added to vocabulary from corpus")

def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)                           # Tensor
    target = torch.LongTensor(batch_size, chunk_len)                        # Tensor
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len-1)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]                                 # chunk still string
        inp[bi] = voc.word_tensor(chunk[:-1])                               # converts to tensor
        
        target[bi] = voc.word_tensor(chunk[1:])                             # converts to tensor
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0
    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()
    perplexities.append(math.exp(loss.item()/args.chunk_len))
    return loss.item() / args.chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

# Initialize models and start training

decoder = CharRNN(
    voc.num_words,
    args.hidden_size,
    voc.num_words,
    model=args.model,
    n_layers=args.n_layers,
)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    decoder.cuda()

start = time.time()
all_losses = []
loss_avg = 0
perplexities = []

try:
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        loss = train(*random_training_set(args.chunk_len, args.batch_size))
        loss_avg += loss

        if epoch % args.print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
            print(generate(decoder, 'I ', args.chunk_len, cuda=args.cuda, voc=voc), '\n')

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

plt.plot(range(1,args.n_epochs+1), perplexities,label='perplexity')
plt.xlabel('Epochs')
plt.ylabel('perplexity')
plt.legend()
plt.show()