import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import config, get_batch, EarlyStopping, plot_losses
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

torch._dynamo.config.suppress_errors = True

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)
        
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(index)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index


def train_bigram_model(train_data, val_data, vocab_size):
    model = BigramLanguageModel(vocab_size).to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    writer = SummaryWriter()

    train_losses = []
    val_losses = []

    max_steps_per_epoch = 1000  # Adjust this value based on your dataset size
    eval_every = 100  # Evaluate less frequently

    for epoch in range(config['epoch']):
        model.train()
        epoch_loss = 0
        steps = 0
        
        pbar = tqdm(total=max_steps_per_epoch, desc=f"Epoch {epoch+1}/{config['epoch']}")
        
        while steps < max_steps_per_epoch:
            xb, yb = get_batch(data, 'train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            pbar.update(1)



        pbar.close()
        print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss/steps:.3f}")

    writer.close()
    plot_losses(train_losses, val_losses)
    return model

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config['eval_iters'])
        for k in range(config['eval_iters']):
            X, Y = get_batch(data, split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == "__main__":
    from utils import load_text_data, create_encoder_decoder, config, get_batch
    config['epoch'] = 10000
    config['learning_rate'] = 5e-3
    all_text = load_text_data()
    vocab = sorted(set(all_text))
    config['vocab_size'] = len(vocab)
    encoder, decoder = create_encoder_decoder(vocab)
    data = torch.tensor(encoder(all_text),dtype=torch.long,device=config['device'])
    train_data, val_data = get_batch(data,'train')
    model = BigramLanguageModel(config['vocab_size'])
    model = model.to(config['device'])
        
        # Initialize the early stopping object
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # Initialize lists to store the losses
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_path = 'best_bigram_model.pth'

    for iter in range(config['epoch']):
        if iter % config['eval_iters'] == 0:
            losses = estimate_loss()
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])

            print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

            # Save the best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path}")

        # Check for early stopping
            early_stopping(losses['val'])
            
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break
            
            
        # sample a batch of data
        xb, yb = get_batch(data, 'train')

        # evaluate the loss
        logits, loss = model.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    # Print the final loss value
    print(f"Final loss: {loss.item():.3f}")
    print(f"Training completed. Best model saved to {best_model_path}")

    # Generate some text using the best model
    model.load_state_dict(torch.load(best_model_path))
    context = torch.zeros((1,1), dtype=torch.long, device=config['device'])
    generated_chars = decoder(model.generate(context, max_new_tokens=500)[0].tolist())
    print("Generated text:")
    print(generated_chars)