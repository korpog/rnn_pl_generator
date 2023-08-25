import random
import torch
import torch.nn as nn
import torchvision.models as models

DATA_PATH = "data/slowa.txt"


def read_lines(file_path):
    with open(file_path, encoding="utf-8") as file:
        return [line.strip() for line in file]
    
def get_unique_chars(li):
    unique_chars = set()
    for word in li:
        for char in word:
            unique_chars.add(char)
    return unique_chars


words = read_lines(DATA_PATH)
all_chars = list(get_unique_chars(words))
n_chars = len(all_chars) + 1 # <EOS> 


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    
# one-hot encoding for input
def input_tensor(line):
    tensor = torch.zeros(len(line), 1, n_chars)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_chars.index(letter)] = 1
    return tensor

def target_tensor(line):
    letter_indexes = [all_chars.index(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_chars - 1) # EOS
    return torch.LongTensor(letter_indexes)


def random_elem(l):
    return l[random.randint(0, len(l) - 1)]

def random_line():
    line = random_elem(words)
    return line

def random_example():
    line = random_line()
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return input_line_tensor, target_line_tensor

criterion = nn.NLLLoss()

learning_rate = 0.0008

rnn = RNN(n_chars, 128, n_chars)

n_iters = 100000
print_every = 10000
all_losses = []

def train(input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.init_hidden()
    rnn.zero_grad()
    loss = torch.Tensor([0])

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)

for iter in range(1, n_iters + 1):
    output, loss = train(*random_example())

    if iter % print_every == 0:
        print(f'Iteration: ({iter} {iter / n_iters * 100:.0f}%) loss: {loss:.2f}')

torch.save(rnn.state_dict(), 'model_weights.pth')

max_length = 15

def sample(start_letter='a'):
    with torch.no_grad():
        input = input_tensor(start_letter)
        hidden = rnn.init_hidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_chars - 1:
                break
            else:
                letter = all_chars[topi]
                output_name += letter
            input = input_tensor(letter)

        return output_name

for _ in range(10):
    rand_char = random_elem(all_chars)
    print(sample(start_letter=rand_char))

