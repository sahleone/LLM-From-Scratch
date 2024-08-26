import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import config
from transformers import GPT2Tokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd
from datasets import Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config['n_embd'], head_size, bias=False)
        self.query = nn.Linear(config['n_embd'], head_size, bias=False)
        self.value = nn.Linear(config['n_embd'], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config['block_size'], config['block_size'])))
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, config['n_embd'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config['dropout']),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, config['n_embd'])
        self.position_embedding_table = nn.Embedding(config['block_size'], config['n_embd'])
        self.blocks = nn.Sequential(*[Block(config['n_embd'], n_head=config['n_head']) for _ in range(config['n_layer'])])
        self.ln_f = nn.LayerNorm(config['n_embd'])
        self.lm_head = nn.Linear(config['n_embd'], vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=config['device']))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config['block_size']:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def train_gpt_model():
    dataset = load_dataset("openwebtext")
    subset_size = 0.01
    num_samples = int(len(dataset['train']) * subset_size)
    dataset = dataset['train'].shuffle(seed=42).select(range(num_samples))

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    config['vocab_size'] = len(tokenizer)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=config['block_size'])

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    train_df = pd.DataFrame(tokenized_datasets)
    train_df, val_test_df = train_test_split(train_df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    def collate_fn(batch):
        return {
            'input_ids': torch.tensor([item['input_ids'] for item in batch], device=config['device']),
            'attention_mask': torch.tensor([item['attention_mask'] for item in batch], device=config['device']),
            'labels': torch.tensor([item['input_ids'] for item in batch], device=config['device'])
        }

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    model = GPTLanguageModel(config['vocab_size']).to(config['device'])
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_dataloader) * config['epoch'])

    writer = SummaryWriter()
    best_val_loss = float('inf')
    best_model_path = 'best_gpt_model.pth'

    def evaluate(dataloader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input_ids']
                labels = batch['labels']
                outputs, loss = model(inputs, targets=labels)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    for epoch in range(config['epoch']):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            inputs = batch['input_ids']
            labels = batch['labels']
            optimizer.zero_grad()
            outputs, loss = model(inputs, targets=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        print(f"Epoch {epoch}, Average Training Loss: {avg_train_loss}")

        val_loss = evaluate(val_dataloader)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        print(f"Epoch {epoch}, Validation Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {val_loss}")

    writer.close()
    print("Training completed.")

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load(best_model_path))
    test_loss = evaluate(test_dataloader)
    print(f"Test Loss: {test_loss}")

    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = train_gpt_model()

    # Generate some text
    context = torch.zeros((1, 1), dtype=torch.long, device=config['device'])
    generated_ids = model.generate(context, max_new_tokens=100)[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    print(generated_text)