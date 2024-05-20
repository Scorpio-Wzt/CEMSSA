import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from datasets import Dataset, DatasetDict, Metric, load_dataset, load_metric
from transformers import (BatchEncoding, EvalPrediction, PreTrainedModel,
                          Trainer, TrainingArguments)
from transformers.tokenization_utils_base import (PaddingStrategy,
                                                  PreTrainedTokenizerBase)
from torch.utils.data import Dataset
from torch.nn.functional import pad

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_tensor = torch.ones_like(target) * self.alpha
            alpha_tensor = alpha_tensor.to(device=input.device, dtype=input.dtype)
            focal_loss = alpha_tensor * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# def preprocess_function(
#     examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizerBase
# ) -> Dict[str, List[str]]:
#     sc_sentences: List[str] = sum([
#         [f'诉方称：{sc}'] * 5 for sc in examples['sc']
#     ], [])

#     bc_sentences: List[str] = sum([
#         [f'辩方回应：{examples[f"bc_{j}"][i]}' for j in range(1, 6)]
#         for i in range(len(examples['id']))
#     ], [])

#     tokenized_examples: BatchEncoding = tokenizer(
#         sc_sentences, bc_sentences, truncation='longest_first', max_length=512
#     )

#     return {k: [v[i:i + 5] for i in range(0, len(v), 5)]
#             for k, v in tokenized_examples.items()}
def cmv_preprocess_function(
    examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizerBase
) -> Dict[str, List[str]]:
    sc_sentences: List[str] = sum([
        [f'Plaintiff say: {sc}'] * 5 for sc in examples['sc']
    ], [])

    bc_sentences: List[str] = sum([
        [f'Defendant say: {examples[f"bc_{j}"][i]}' for j in range(1, 6)]
        for i in range(len(examples['id']))
    ], [])

    sc_contexts: List[str] = sum([
        [sc_context] * 5 for sc_context in examples['sc_context']
    ], [])

    bc_contexts: List[str] = sum([
        [f'Defendant say: {examples[f"bc_{j}_context"][i]}' for j in range(1, 6)]
        for i in range(len(examples['id']))
    ], [])

    # combined_sentences = [
    # f'{sc} [SEP] {bc} [SEP] {bc_context}'
    # for sc, bc, bc_context in zip(sc_sentences, bc_sentences, bc_contexts)
    # ]
    combined_sentences = [
    f'{sc} [SEP] {bc}'
    for sc, bc in zip(sc_sentences, bc_sentences)
    ]

    tokenized_examples: BatchEncoding = tokenizer(
    combined_sentences, truncation='longest_first', max_length=512
    )

    return {k: [v[i:i + 5] for i in range(0, len(v), 5)]
            for k, v in tokenized_examples.items()}
def preprocess_function(
    examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizerBase
) -> Dict[str, List[str]]:
    sc_sentences: List[str] = sum([
        [f'诉方称：{sc}'] * 5 for sc in examples['sc']
    ], [])

    bc_sentences: List[str] = sum([
        [f'辩方回应：{examples[f"bc_{j}"][i]}' for j in range(1, 6)]
        for i in range(len(examples['id']))
    ], [])

    context_sentences: List[str] = sum([
        [context] * 5 for context in examples['context']
    ], [])

    combined_sentences = [
    f'{sc} [SEP] {bc} [SEP] {context}'
    for sc, bc, context in zip(sc_sentences, bc_sentences, context_sentences)
    ]

    tokenized_examples: BatchEncoding = tokenizer(
    combined_sentences, truncation='longest_first', max_length=512
    )
    # tokenized_examples: BatchEncoding = tokenizer(
    # sc_sentences, bc_sentences, truncation='longest_first', max_length=512
    # )
    
    return {k: [v[i:i + 5] for i in range(0, len(v), 5)]
            for k, v in tokenized_examples.items()}

def get_cmv_dataset(
    location: str, tokenizer: PreTrainedTokenizerBase
) -> Union[Dataset, DatasetDict]:
    dataset: Dataset = load_dataset('json', data_files=location, split='train')

    dataset = dataset.map(
        lambda x: cmv_preprocess_function(x, tokenizer), batched=True
    )
        
    if 'answer' in dataset.column_names:
        return dataset.rename_column(
            'answer', 'labels'
        ).train_test_split(test_size=0.1)
    else:
        return dataset
    
def get_dataset(
    location: str, tokenizer: PreTrainedTokenizerBase
) -> Union[Dataset, DatasetDict]:
    dataset: Dataset = load_dataset('json', data_files=location, split='train')

    dataset = dataset.remove_columns(
        ['text_id', 'category', 'chapter', 'crime']
    ).map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )
        
    if 'answer' in dataset.column_names:
        return dataset.rename_column(
            'answer', 'labels'
        ).train_test_split(test_size=0.1)
    else:
        return dataset

def get_cl_dataset(
    positive_pairs: List[tuple], negative_pairs: List[tuple], tokenizer: PreTrainedTokenizerBase
) -> Dataset:
    # 对每个元组进行处理并创建样本对
    tokenized_positive_examples = tokenizer(
        [item[0] for item in positive_pairs],  # 提取正样本中的第一个文本
        truncation='longest_first',
        max_length=512
    )

    tokenized_negative_examples = tokenizer(
        [item[0] for item in negative_pairs],  # 提取负样本中的第一个文本
        truncation='longest_first',
        max_length=512
    )

    return tokenized_positive_examples, tokenized_negative_examples



def get_all_dataset(
    location: str, tokenizer: PreTrainedTokenizerBase
) -> Union[Dataset, DatasetDict]:
    dataset: Dataset = load_dataset('json', data_files=location, split='train')

    dataset = dataset.remove_columns(
        ['text_id', 'category', 'chapter', 'crime']
    ).map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )

    if 'answer' in dataset.column_names:
        return dataset.rename_column(
            'answer', 'labels'
        )
    else:
        return dataset

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])

        if 'labels' in features[0].keys():
            labels: Optional[List[int]] = [
                feature.pop('labels') for feature in features
            ]
        else:
            labels = None

        flattened_features: List[Dict[str, Any]] = sum([
            [{k: v[i] for k, v in feature.items()}
             for i in range(num_choices)] for feature in features
        ], [])

        padded_features: BatchEncoding = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )

        batch: Dict[str, torch.Tensor] = {
            k: v.view(batch_size, num_choices, -1)
            for k, v in padded_features.items()
        }

        if labels is not None:
            batch['labels'] = torch.tensor(labels, dtype=torch.int64) - 1
        return batch

def dynamic_collate_fn_cl(batch):
    # 提取每个样本的 'input_ids'、'attention_mask' 、'token_type_ids'和 'labels'
    anchor_input_ids = [item['anchor_input_ids'] for item in batch]
    anchor_attention_mask = [item['anchor_attention_mask'] for item in batch]
    anchor_token_type_ids = [item['anchor_token_type_ids'] for item in batch]
    positive_input_ids = [item['positive_input_ids'] for item in batch]
    positive_attention_mask = [item['positive_attention_mask'] for item in batch]
    positive_token_type_ids = [item['positive_token_type_ids'] for item in batch]
    negative1_input_ids = [item['negative1_input_ids'] for item in batch]
    negative1_attention_mask = [item['negative1_attention_mask'] for item in batch]
    negative1_token_type_ids = [item['negative1_token_type_ids'] for item in batch]
    negative2_input_ids = [item['negative2_input_ids'] for item in batch]
    negative2_attention_mask = [item['negative2_attention_mask'] for item in batch]
    negative2_token_type_ids = [item['negative2_token_type_ids'] for item in batch]
    negative3_input_ids = [item['negative3_input_ids'] for item in batch]
    negative3_attention_mask = [item['negative3_attention_mask'] for item in batch]
    negative3_token_type_ids = [item['negative3_token_type_ids'] for item in batch]
    negative4_input_ids = [item['negative4_input_ids'] for item in batch]
    negative4_attention_mask = [item['negative4_attention_mask'] for item in batch]
    negative4_token_type_ids = [item['negative4_token_type_ids'] for item in batch]

    anchor_input_ids = torch.tensor(anchor_input_ids, dtype=torch.long)

    anchor_attention_mask = torch.tensor(anchor_attention_mask, dtype=torch.long)

    anchor_token_type_ids = torch.tensor(anchor_token_type_ids, dtype=torch.long)
 
    positive_input_ids = torch.tensor(positive_input_ids, dtype=torch.long)

    positive_attention_mask = torch.tensor(positive_attention_mask, dtype=torch.long)

    positive_token_type_ids = torch.tensor(positive_token_type_ids, dtype=torch.long)
 
    negative1_input_ids = torch.tensor(negative1_input_ids, dtype=torch.long)

    negative1_attention_mask = torch.tensor(negative1_attention_mask, dtype=torch.long)

    negative1_token_type_ids = torch.tensor(negative1_token_type_ids, dtype=torch.long)

    negative2_input_ids = torch.tensor(negative2_input_ids, dtype=torch.long)

    negative2_attention_mask = torch.tensor(negative2_attention_mask, dtype=torch.long)

    negative2_token_type_ids = torch.tensor(negative2_token_type_ids, dtype=torch.long)

    negative3_input_ids = torch.tensor(negative3_input_ids, dtype=torch.long)

    negative3_attention_mask = torch.tensor(negative3_attention_mask, dtype=torch.long)

    negative3_token_type_ids = torch.tensor(negative3_token_type_ids, dtype=torch.long)

    negative4_input_ids = torch.tensor(negative4_input_ids, dtype=torch.long)

    negative4_attention_mask = torch.tensor(negative4_attention_mask, dtype=torch.long)

    negative4_token_type_ids = torch.tensor(negative4_token_type_ids, dtype=torch.long)
    return {
        'anchor_input_ids': anchor_input_ids,
        'anchor_attention_mask': anchor_attention_mask,
        'anchor_token_type_ids':anchor_token_type_ids,
        'positive_input_ids': positive_input_ids,
        'positive_attention_mask': positive_attention_mask,
        'positive_token_type_ids':positive_token_type_ids,
        'negative1_input_ids': negative1_input_ids,
        'negative1_attention_mask': negative1_attention_mask,
        'negative1_token_type_ids':negative1_token_type_ids,
        'negative2_input_ids': negative2_input_ids,
        'negative2_attention_mask': negative2_attention_mask,
        'negative2_token_type_ids':negative2_token_type_ids,
        'negative3_input_ids': negative3_input_ids,
        'negative3_attention_mask': negative3_attention_mask,
        'negative3_token_type_ids':negative3_token_type_ids,
        'negative4_input_ids': negative4_input_ids,
        'negative4_attention_mask': negative4_attention_mask,
        'negative4_token_type_ids':negative4_token_type_ids,
    }

def dynamic_collate_fn_test(batch):
    # 提取每个样本的 'input_ids'、'attention_mask' 、'token_type_ids'和 'labels'
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    
    # 计算批次中最大的序列长度
    max_seq_length = max(len(sentence) for seq in input_ids for sentence in seq)
    
    # 填充 input_ids
    padded_input_ids = []
    for seq in input_ids:
        padded_seq = []
        for sentence in seq:
            padded_sentence = sentence + [0] * (max_seq_length - len(sentence))
            padded_seq.append(padded_sentence)
        padded_input_ids.append(padded_seq)
    
    # 填充 attention_mask
    padded_attention_mask = []
    for seq in attention_mask:
        padded_seq = []
        for sentence in seq:
            padded_sentence = sentence + [0] * (max_seq_length - len(sentence))
            padded_seq.append(padded_sentence)
        padded_attention_mask.append(padded_seq)
        
    # 填充 token_type_ids
    padded_token_type_ids = []
    for seq in token_type_ids:
        padded_seq = []
        for sentence in seq:
            padded_sentence = sentence + [0] * (max_seq_length - len(sentence))
            padded_seq.append(padded_sentence)
        padded_token_type_ids.append(padded_seq)
    
    # 转换为张量
    padded_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    padded_attention_mask = torch.tensor(padded_attention_mask, dtype=torch.long)
    padded_token_type_ids = torch.tensor(padded_token_type_ids, dtype=torch.long)
    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'token_type_ids':padded_token_type_ids,
    }
    
def dynamic_collate_fn(batch):
    # 提取每个样本的 'input_ids'、'attention_mask' 、'token_type_ids'和 'labels'
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # 计算批次中最大的序列长度
    max_seq_length = max(len(sentence) for seq in input_ids for sentence in seq)
    
    # 填充 input_ids
    padded_input_ids = []
    for seq in input_ids:
        padded_seq = []
        for sentence in seq:
            padded_sentence = sentence + [0] * (max_seq_length - len(sentence))
            padded_seq.append(padded_sentence)
        padded_input_ids.append(padded_seq)
    
    # 填充 attention_mask
    padded_attention_mask = []
    for seq in attention_mask:
        padded_seq = []
        for sentence in seq:
            padded_sentence = sentence + [0] * (max_seq_length - len(sentence))
            padded_seq.append(padded_sentence)
        padded_attention_mask.append(padded_seq)
        
    # 填充 token_type_ids
    padded_token_type_ids = []
    for seq in token_type_ids:
        padded_seq = []
        for sentence in seq:
            padded_sentence = sentence + [0] * (max_seq_length - len(sentence))
            padded_seq.append(padded_sentence)
        padded_token_type_ids.append(padded_seq)
    
    # 转换为张量
    padded_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    padded_attention_mask = torch.tensor(padded_attention_mask, dtype=torch.long)
    padded_token_type_ids = torch.tensor(padded_token_type_ids, dtype=torch.long)
    labels = torch.tensor(labels)
    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'token_type_ids':padded_token_type_ids,
        'labels': labels
    }

def dynamic_collate_fn_bert(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # 将嵌套的句子序列连接在一起
    flat_input_ids = [sentence for seq in input_ids for sentence in seq]
    flat_attention_mask = [sentence for seq in attention_mask for sentence in seq]
    flat_token_type_ids = [sentence for seq in token_type_ids for sentence in seq]

    # 计算批次中最大的序列长度
    max_seq_length = max(len(sentence) for sentence in flat_input_ids)

    # 填充 input_ids 和生成 attention_mask
    padded_input_ids = []
    padded_attention_mask = []
    for sentence in flat_input_ids:
        padded_sentence = sentence + [0] * (max_seq_length - len(sentence))
        padded_input_ids.append(padded_sentence)

    for sentence in flat_attention_mask:
        attention_mask_sentence = sentence + [0] * (max_seq_length - len(sentence))
        padded_attention_mask.append(attention_mask_sentence)

    # 填充 token_type_ids
    padded_token_type_ids = []
    for sentence in flat_token_type_ids:
        padded_sentence = sentence + [0] * (max_seq_length - len(sentence))
        padded_token_type_ids.append(padded_sentence)

    # 转换为张量
    padded_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    padded_attention_mask = torch.tensor(padded_attention_mask, dtype=torch.long)
    padded_token_type_ids = torch.tensor(padded_token_type_ids, dtype=torch.long)
    labels = torch.tensor(labels)

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'token_type_ids': padded_token_type_ids,
        'labels': labels
    }

def dynamic_collate_fn_berttest(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]

    # 将嵌套的句子序列连接在一起
    flat_input_ids = [sentence for seq in input_ids for sentence in seq]
    flat_attention_mask = [sentence for seq in attention_mask for sentence in seq]
    flat_token_type_ids = [sentence for seq in token_type_ids for sentence in seq]

    # 计算批次中最大的序列长度
    max_seq_length = max(len(sentence) for sentence in flat_input_ids)

    # 填充 input_ids 和生成 attention_mask
    padded_input_ids = []
    padded_attention_mask = []
    for sentence in flat_input_ids:
        padded_sentence = sentence + [0] * (max_seq_length - len(sentence))
        padded_input_ids.append(padded_sentence)

    for sentence in flat_attention_mask:
        attention_mask_sentence = sentence + [0] * (max_seq_length - len(sentence))
        padded_attention_mask.append(attention_mask_sentence)

    # 填充 token_type_ids
    padded_token_type_ids = []
    for sentence in flat_token_type_ids:
        padded_sentence = sentence + [0] * (max_seq_length - len(sentence))
        padded_token_type_ids.append(padded_sentence)

    # 转换为张量
    padded_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    padded_attention_mask = torch.tensor(padded_attention_mask, dtype=torch.long)
    padded_token_type_ids = torch.tensor(padded_token_type_ids, dtype=torch.long)

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'token_type_ids': padded_token_type_ids,
    }


def compute_metrics(
    eval_pred: EvalPrediction, metric: Metric
) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print("predictions", predictions)
    return metric.compute(predictions=predictions, references=labels)

def get_trainer(
    train_set: Optional[Dataset], test_set: Optional[Dataset],
    tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel
) -> Trainer:
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        weight_decay=0.01,
        # no_cuda = True,
        no_cuda=not torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy',
        greater_is_better=True,
        optim='adamw_torch',
        report_to='none'
    )

    metric = load_metric('accuracy')

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=test_set,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=lambda x: compute_metrics(x, metric)
    )

class AutomaticWeightedLoss(torch.nn.Module):
    """
    Automatically weighted multi-task loss.

    Params:
        num: int, the number of loss
        x: multi-task loss

    Examples:
        loss1 = 1
        loss2 = 2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            try:
                part1 = 0.5 / (self.params[i] ** 2) * loss
                part2 = F.softplus(self.params[i] ** 2)
                loss_sum += part1 + part2

            except ZeroDivisionError:

                pass
        return loss_sum

