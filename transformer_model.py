import torch
import torch.nn as nn
from torch.nn import functional as F
from bs4 import BeautifulSoup
import os
from langdetect import detect
import re
import time

with open('discord_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

characters = sorted(list(set(text)))
vocab_size = len(characters)

string2int = {char:index for index, char in enumerate(characters)} # dictionary mapping a string to integer
int2string = {index:char for index, char in enumerate(characters)} #dictionary mapping integer to string
def char_encoder(s):
  # input s is a string.
  # output is a list of integers mapping from their respective character.
  return [string2int[char] for char in s]

def int_decoder(i):
  # input i is a list of integers
  # output is a string
  lst_str = [int2string[num] for num in i]
  output = ''.join(lst_str)
  return output

data = torch.tensor(char_encoder(text))
n = int(0.9*len(data)) # 90% train
train_data = data[:n]
test_data = data[n:]

n_heads = 4
n_layers = 6
max_iters = 5000
n_embedding = 128
batch_size = 64
context_length =  64
learning_rate = 3e-4

def batches(n_batches, split):
  if split == 'train': # split whether training or testing
    data = train_data
  elif split == 'test':
    data = test_data
  ind = torch.randint(len(data) - context_length, (batch_size,)) # grabbing random starting characters index. subtract the context length to ensure that every batch has enough characters.
  x = torch.stack([data[i:i+context_length] for i in ind]) # getting the character from the index and getting up to context length. for every batch and stacking on the batch dimension.
  y = torch.stack([data[i+1:i+1+context_length] for i in ind]) # the target is the next character, so we index by adding 1.
  return x,y

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.query = nn.Linear(n_embedding, head_size)
    self.key = nn.Linear(n_embedding, head_size)
    self.value = nn.Linear(n_embedding, head_size)
    self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length))) # this tells pytorch that this is a non trainable parameter, but needed for the forward pass
  def forward(self,x):
    B, L, H = x.shape # Batch size, Sequence length, Hidden size (embeddings)
    key = self.key(x)
    query = self.query(x)
    value = self.value(x)
    attn = query @ key.transpose(1,2) * (1/H**0.5)
    attn = attn.masked_fill(self.tril[:L, :L] == 0, float('-inf')) # masked self attention
    attn = F.softmax(attn, dim = -1)
    attn = attn @ value
    return attn
  
class multiHead(nn.Module):
  def __init__(self, n_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
    self.projection = nn.Linear(n_embedding, n_embedding) # projection layer for residual pathways
  def forward(self,x):
    stackedheads = torch.cat([head(x) for head in self.heads], dim = -1)
    out = self.projection(stackedheads)
    return out
  
class Block(nn.Module):
  def __init__(self, n_embedding, n_heads):
    super().__init__()
    head_size = n_embedding // n_heads
    self.attention = multiHead(n_heads, head_size)
    self.layernorm1 = nn.LayerNorm(n_embedding)
    self.layernorm2 = nn.LayerNorm(n_embedding)
    self.network = nn.Sequential(
        nn.Linear(n_embedding, 4 * n_embedding), # from transformer literature, the inner layer of the FF network should be 4 times the input/output layer.
        nn.GELU(),
        nn.Linear(4 * n_embedding, n_embedding) # projection layer for residual pathways
    )
  def forward(self, x):
    x = self.layernorm1(x) # layer norm before the transformation (this is different from the paper, but this is done more commonly now)
    x = x + self.attention(x) # adds residual connection
    x = self.layernorm2(x)
    x = x + self.network(x) # adds residual connection
    return x
  
class decoderTransformer(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding = nn.Embedding(vocab_size, n_embedding)
    self.position_embedding = nn.Embedding(context_length, n_embedding)
    self.blocks = nn.Sequential(*[Block(n_embedding = n_embedding, n_heads=n_heads) for _ in range(n_layers)])
    self.layer_norm3 = nn.LayerNorm(n_embedding) # final layer norm in architecture before feeding into MLP
    self.final_linear = nn.Linear(n_embedding, vocab_size) # final layer for logits
  def forward(self,x, targets = None):
    B, S = x.shape
    t_embed = self.token_embedding(x)
    p_embed = self.position_embedding(torch.arange(S)) # each position is assigned to an integer for position. torch.arrange create 1d tensor from 0 to S-1. One embedding vector for each context position.
    x = t_embed + p_embed
    x = self.blocks(x)
    x = self.layer_norm3(x)
    logits = self.final_linear(x)
    if targets is None: # for generation, during that we do not give a loss.
      logloss = None
    else: # for training
      B, S, H = logits.shape
      logits = logits.view(B*S, H) # want to stretch your batches arcoss the sequence dimension to get a large 2d matrix that has every sequence and its embedding.
      targets = targets.view(B*S) # since the logits are stretcted, do the same for the targets/labels and make it one dimensional.
      logloss = F.cross_entropy(logits, targets)
    return logits, logloss

  def generate(self, x, max_tokens):
    for _ in range(max_tokens):
      x_cropped = x[:,-context_length:]
      logits, loss = self(x_cropped)
      logits = logits[:,-1,:] # only get the last context.
      probability = F.softmax(logits, dim = 1)
      next_token = torch.multinomial(probability, num_samples = 1)
      x = torch.cat((x, next_token), dim = 1)
    return x
  
  # for plotting reasons
steps_plot = []
train_loss_plt = []
test_loss_plt = []

model = decoderTransformer()
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

start_time=time.time()
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    # if iter % 100 == 0 or iter == max_iters - 1:
    #     losses = loss.item()
    #     print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
  xb, yb = batches(batch_size, 'train')

    # evaluate the loss
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

  if iter % 100 == 0 or iter == max_iters - 1:
    steps_plot.append(iter)
    # need to get both train and test loss.
    # to do this, i need to sample batches from both splits
    train_loss = []
    test_loss = []
    model.eval() # puts model in evaluation mode
    for split in ['train', 'test']:
      x_,y_ = batches(batch_size, split)
      logits, loss = model(x_,y_)
      if split == 'test':
        test_loss_plt.append(loss.item())
        test_loss.append(loss.item())
      else:
        train_loss_plt.append(loss.item())
        train_loss.append(loss.item())
    model.train() # back to training mode

    losses = loss.item()
    print(f"step {iter}: train loss {train_loss[-1]:.4f}, val loss {test_loss[-1]:.4f}") # may not work
end_time=time.time()
    # print(f"step {iter}: train loss {losses:.4f}, val loss {losses:.4f}")

train_time=(end_time-start_time)/60/60
print(f'Training time: {train_time:.4f} hours.')

starting = torch.zeros((1,1), dtype = torch.long)
model_generate = model.generate(starting, max_tokens = 1000)[0].tolist()
print(int_decoder(model_generate))

with open(f'training_{(sum(p.numel() for p in model.parameters())/1e6):.0f}M_error.txt', 'w') as file:
    # Write the string list to the file
    for item in train_loss_plt:
        file.write(str(item)+'\n')
with open(f'testing_{(sum(p.numel() for p in model.parameters())/1e6):.0f}M_error.txt', 'w') as file:
    # Write the string list to the file
    for item in test_loss_plt:
        file.write(str(item) + '\n')

hparam_dict = {
'n_heads':n_heads,
'n_layers':n_layers,
'max_iters':max_iters,
'n_embedding':n_embedding,
'batch_size':batch_size,
'context_length':context_length,
'learning_rate':learning_rate}

with open(f'{(sum(p.numel() for p in model.parameters())/1e6):.0f}M_hparams.txt', 'w') as f:
  for key,value in hparam_dict.items():
    f.write(f'{key}:{str(value)}' + '\n')
