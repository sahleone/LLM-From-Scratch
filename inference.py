import argparse
import torch
import os
from BigramLanguageModel import BigramLanguageModel
from GPTLanguageModel import GPTLanguageModel
from utils import config, create_encoder_decoder

def load_bigram_model(model_path, vocab_size):
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    # Load the state dict first
    state_dict = torch.load(model_path, map_location=config['device'])
    
    # Get the correct vocab size from the loaded state dict
    correct_vocab_size = state_dict['token_embedding_table.weight'].shape[0]
    
    # Create the model with the correct vocab size
    model = BigramLanguageModel(correct_vocab_size).to(config['device'])
    
    # Load the state dict
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_gpt_model(model_path, vocab_size):
    # Load the configuration from the saved model
    config = torch.load(model_path, map_location=config['device'])['config']
    
    # Create the model with the loaded configuration
    model = GPTLanguageModel(config)
    
    # Load the state dict
    model.load_state_dict(torch.load(model_path, map_location=config['device'])['state_dict'])
    
    return model

def generate_text_bigram(model, decoder, max_new_tokens=100):
    context = torch.zeros((1, 1), dtype=torch.long, device=config['device'])
    generated_ids = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    return decoder(generated_ids)

def generate_text_gpt(model, tokenizer, max_new_tokens=100):
    context = torch.zeros((1, 1), dtype=torch.long, device=config['device'])
    generated_ids = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    return tokenizer.decode(generated_ids)

def main(args):
    # Load vocabulary and create encoder/decoder
    with open(args.vocab_file, 'r') as f:
        vocab = f.read().splitlines()
    
    encoder, decoder = create_encoder_decoder(vocab)
    config['vocab_size'] = len(vocab)

    # Update config with command-line arguments
    config['device'] = args.device
    
    print(f"Using device: {config['device']}")
    print(f"Vocabulary size: {config['vocab_size']}")

    if args.model == 'bigram' or args.model == 'both':
        # Load Bigram model
        bigram_model = load_bigram_model(args.bigram_model, config['vocab_size'])
        if bigram_model is None:
            print("Failed to load Bigram model. Skipping Bigram inference.")
        else:
            print(f"Loaded Bigram model with vocab size: {bigram_model.token_embedding_table.weight.shape[0]}")
            # Generate text using Bigram model
            print("Generating text using Bigram model:")
            bigram_text = generate_text_bigram(bigram_model, decoder, args.max_tokens)
            print(bigram_text)
            print("\n" + "="*50 + "\n")

    if args.model == 'gpt' or args.model == 'both':
        # Load GPT model
        gpt_model = load_gpt_model(args.gpt_model, config['vocab_size'])

        # Generate text using GPT model
        print("Generating text using GPT model:")
        gpt_text = generate_text_gpt(gpt_model, decoder, args.max_tokens)
        print(gpt_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on trained language models.")
    parser.add_argument('--model', choices=['bigram', 'gpt', 'both'], default='both',
                        help="Choose which model to use for inference")
    parser.add_argument('--bigram-model', default='best_bigram_model.pth',
                        help="Path to the trained Bigram model")
    parser.add_argument('--gpt-model', default='best_gpt_model.pth',
                        help="Path to the trained GPT model")
    parser.add_argument('--vocab-file', default='vocab.txt',
                        help="Path to the vocabulary file")
    parser.add_argument('--max-tokens', type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu',
                        help="Device to run inference on")

    args = parser.parse_args()
    main(args)