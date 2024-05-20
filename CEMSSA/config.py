import argparse

def get_config():
    config = argparse.ArgumentParser()
    config.add_argument("--vocab_file_path", default="bert-base-chinese/vocab.txt", type=str)
    config.add_argument("--model_path", default="bert-base-chinese/", type=str)
    config.add_argument("--train_path", default="data/processed_data.jsonl", type=str)
    config.add_argument("--batch-size", default=1, type=int)
    config.add_argument("--num_train_epochs", default=10, type=int)
    config.add_argument("--learning_rate", default=2e-5, type=float)
    config.add_argument("--weight_decay", default=0.01, type=float)
    config.add_argument("--gradient_accumulation_steps", default=4, type=int)
    config.add_argument("--random_seed", default=42, type=int)
    config.add_argument("--epsilon", default=0.4, type=float)
    config.add_argument("--num_classes", default=5, type=int)
    config.add_argument("--K", default=3, type=int)
    config.add_argument("--alpha", default=0.3, type=float)
    config.add_argument("--lossAlpha", default=0.3, type=float)
    config.add_argument("--smoothing", default=0.1, type=float)
    config.add_argument("--temperature", default=0.5, type=float)
    
    config = config.parse_args()

    return config