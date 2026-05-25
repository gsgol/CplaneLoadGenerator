import torch
import json
from pathlib import Path
from typing import List
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

from config import PROC_LIST, ALLOWED_NEXT, DEFAULT_MODEL_NAME
from data_loader import SequenceDataLoader, load_multiple_files, save_sequences


def prepare_dataset(
    sequences: List[List[str]],
    tokenizer,
    max_length: int = 1024,
) -> Dataset:
    texts = [" ".join(seq) for seq in sequences]

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None,
    )

    dataset_dict = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
    }

    return Dataset.from_dict(dataset_dict)


def train_model(
    input_data_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str = "./trained_model",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 5e-5,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    logging_steps: int = 100,
    save_steps: int = 1000,
    eval_steps: int = 500,
    eval_split: float = 0.1,
    seed: int = 42,
    use_fp16: bool = False,
    validate_data: bool = True,
    filter_invalid: bool = True,
):
    torch.manual_seed(seed)
    loader = SequenceDataLoader(input_data_path)
    sequences = loader.load()
    stats = loader.get_statistics()

    for proc, count in sorted(stats['procedure_counts'].items()):
        percentage = (count / stats['total_procedures']) * 100
        print(f"{proc:12}: {count:6d} ({percentage:5.2f}%)")

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    new_tokens = [name for name in PROC_LIST if name not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))

    dataset = prepare_dataset(sequences, tokenizer)

    split_idx = int(len(dataset) * (1 - eval_split))
    train_dataset = dataset.select(range(split_idx))
    eval_dataset = dataset.select(range(split_idx, len(dataset)))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    logging_dir=f"{output_dir}/logs",
    logging_steps=logging_steps,
    save_steps=eval_steps,
    eval_steps=eval_steps,
    save_strategy="steps",
    eval_strategy="steps",
    save_total_limit=3,
    fp16=use_fp16 and torch.cuda.is_available(),
    gradient_accumulation_steps=1,
    dataloader_num_workers=4,
    seed=seed,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    training_info = {
        "input_data_path": input_data_path,
        "base_model": model_name,
        "total_sequences": len(sequences),
        "train_sequences": len(train_dataset),
        "eval_sequences": len(eval_dataset),
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "batch_size": per_device_train_batch_size,
        "statistics": stats,
    }

    info_path = Path(output_dir) / "training_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)

    print(f"Модель сохранена в {output_dir}")

    return model, tokenizer


def train_from_multiple_files(
    file_paths: List[str],
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str = "./trained_model",
    **kwargs
):
    all_sequences = load_multiple_files(file_paths)

    temp_path = "./data/temp_combined.json"
    save_sequences(all_sequences, temp_path)

    model, tokenizer = train_model(
        input_data_path=temp_path,
        model_name=model_name,
        output_dir=output_dir,
        **kwargs,
    )

    Path(temp_path).unlink()
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = train_model(
        input_data_path="./data/input_sequences.json",
        model_name=DEFAULT_MODEL_NAME,
        output_dir="./trained_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        seed=42,
        validate_data=True,
        filter_invalid=True,
    )