import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
from datasets import DatasetDict
from transformers import (BertForMultipleChoice, BertTokenizer, Trainer, set_seed)
from util import get_dataset, get_trainer, get_cmv_dataset

def main():
    set_seed(42)
    vocab_file_path = 'bert-base-uncased/vocab.txt'
    tokenizer: BertTokenizer = BertTokenizer(vocab_file_path)
    model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
    split_set: DatasetDict = get_cmv_dataset('datasets/cmv/trainandvalid.jsonl', tokenizer)
    # split_set: DatasetDict = get_dataset('data/chinese_train_entry.jsonl', tokenizer)
    trainer: Trainer = get_trainer(split_set['train'], split_set['test'],
                                tokenizer, model)

    trainer.train()
    tokenizer.save_pretrained('model/cmvMC')# baseFgmCalrProcess
    model.save_pretrained('model/cmvMC')

if __name__ == '__main__':
    main()