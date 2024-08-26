# Building Language Models from Scratch

## Overview

This project is a work in progress inspired by freeCodeCamp and Andrej Karpathy's teachings on building language models from scratch. It aims to implement and train both a simple bigram model and a more complex GPT-like model using PyTorch.

## Project Structure

- `LLM.ipynb`: Main Jupyter notebook containing the implementation of both bigram and GPT models.
- `llmScript.py`: Python script version of the notebook for easier execution and integration.
- `utils.py`: Utility functions for data loading, preprocessing, and model evaluation.
- `requirements.txt`: List of Python dependencies required for the project.

## Features

- Implementation of a bigram language model
- Implementation of a GPT-like transformer model
- Custom dataset loading and preprocessing
- Training and evaluation loops
- Text generation capabilities

## TODO

1. Split `LLM.ipynb` into two separate notebooks:
   - One for the bigram model
   - One for the GPT model
2. Implement TensorBoard for better visualization of training progress
3. Convert Jupyter notebooks to Python scripts for improved modularity
4. Extend training duration for better model performance
5. Experiment with training on specific domains (e.g., scientific papers, literature)
6. Implement more advanced tokenization techniques
7. Add model checkpointing and resumable training
8. Explore different optimization techniques and hyperparameters
9. Create my own tokenizer

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebooks or Python scripts

## Contributing

This project is a work in progress, and contributions are welcome. Please feel free to submit issues or pull requests with improvements or suggestions.

## Acknowledgements

- freeCodeCamp for their excellent tutorials on building language models
- Andrej Karpathy for his insightful lectures and implementations of transformer models

## License

[MIT License](LICENSE)