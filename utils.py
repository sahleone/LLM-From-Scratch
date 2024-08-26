import torch
import os
from matplotlib import pyplot as plt

# Global configurations
config = {
    'block_size': 64,
    'batch_size': 128,
    'epoch': 10,
    'learning_rate': 5e-5,
    'eval_iters': 250,
    'n_embd': 512,
    'n_head': 4,
    'n_layer': 4,
    'dropout': 0.2,
    'device': torch.device('cpu')  # default device
}

# Set device based on availability
if torch.cuda.is_available():
    config['device'] = torch.device('cuda')
    print("Using CUDA (GPU)")
elif torch.backends.mps.is_available():
    config['device'] = torch.device('mps')
    print("Using MPS")
else:
    print("Using CPU")

print("Device:", config['device'])

# Load and preprocess text data
def load_text_data(books_dir='books'):
    all_text = ""
    for filename in os.listdir(books_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(books_dir, filename), 'r', encoding='utf-8') as f:
                all_text += f.read() + " "
    return all_text

# Encoding and decoding functions
def create_encoder_decoder(vocab):
    string_to_int = {ch: i for i, ch in enumerate(vocab)}
    int_to_string = {i: ch for i, ch in enumerate(vocab)}
    
    def encoder(string):
        return [string_to_int[ch] for ch in string]
    
    def decoder(vector):
        return "".join([int_to_string.get(i, '<UNK>') for i in vector])
    
    return encoder, decoder

# Batch generation function
def get_batch(data, split):
    n = len(data)
    split_index = int(n * 0.8)
    train_data, val_data = data[:split_index], data[split_index:]
    batch_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(batch_data) - config['block_size'], (config['batch_size'],))
    x = torch.stack([batch_data[i:i+config['block_size']] for i in ix])
    y = torch.stack([batch_data[i+1:i+config['block_size']+1] for i in ix])
    return x.to(config['device']), y.to(config['device'])

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Plot training and validation losses
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Evaluation Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.show()