### 1. 데이터 및 환경설정 ###
import os
import re
from datetime import datetime
from pprint import pprint

import pandas as pd
import torch
import wandb
import yaml
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments

load_dotenv()
TEAM = os.getenv("TEAM")
OS_PATH = os.getenv("OS_PATH")

file_name = "_".join(os.path.splitext(os.path.basename(__file__))[0].split("_")[:3])
OUTPUT_PATH = os.path.join(OS_PATH, "output", file_name)
DATA_PATH = os.path.join(OS_PATH, "data")

config_name = os.path.join(OS_PATH, "config", f"{file_name}.yaml")
if os.path.exists(config_name):
    with open(config_name, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
else:
    raise FileNotFoundError(f"{config_name} 파일이 존재하지 않습니다!")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained(config["general"]["model_name"])

exp_name = file_name.removeprefix("nlp_ds_")
now = datetime.now().strftime("%m%d-%H%M")
config["general"]["exp_name"] = f"{exp_name}_{now}"
config["general"]["output_dir"] = OUTPUT_PATH
config["wandb"]["entity"] = TEAM
config["wandb"]["dir"] = os.path.join(OUTPUT_PATH, "logs")

config["tokenizer"]["bos_token"] = tokenizer.bos_token
config["tokenizer"]["eos_token"] = tokenizer.eos_token

wandb.init(
    entity=config["wandb"]["entity"],
    project=config["wandb"]["project"],
    name=config["general"]["exp_name"],
    dir=config["wandb"]["dir"],
)

pprint(config)
print("*" * 10, config["general"]["exp_name"], "*" * 10)


### 2. 데이터 가공 및 데이터셋 클래스 구축 ###
class Preprocess:
    def __init__(self, bos_token: str, eos_token: str) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token

    def clean_dialogue(self, text: str) -> str:
        text = re.sub(r"#Person(\d+)#\s*(?::)?\s*", r"P\1: ", text)
        text = re.sub(r"[^\S\r\n]+", " ", text)
        text = re.sub(r"(?<!\n)(P\d+:)", r"\n\1", text).strip()
        return text

class SummaryDataset(Dataset):
    def __init__(self, df, tokenizer, config):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        instruction = "다음 대화를 요약하세요."
        dialogue_part = f"### User:\n{instruction}\n\n### 대화:\n{item['dialogue']}\n\n### Assistant:\n"
        summary_part = f"{item['summary']}{self.tokenizer.eos_token}"

        full_text = dialogue_part + summary_part

        encodings = self.tokenizer(
            full_text,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze()
        labels = input_ids.clone()

        prompt_tokenized = self.tokenizer(dialogue_part, truncation=True, max_length=1024)
        prompt_len = len(prompt_tokenized["input_ids"])

        labels[:prompt_len] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": labels,
        }

    def __len__(self):
        return len(self.df)

def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
    train_file = os.path.join(data_path, "train.csv")
    val_file = os.path.join(data_path, "dev.csv")

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    train_df["dialogue"] = train_df["dialogue"].apply(preprocessor.clean_dialogue)
    val_df["dialogue"] = val_df["dialogue"].apply(preprocessor.clean_dialogue)

    return SummaryDataset(train_df, tokenizer, config), SummaryDataset(val_df, tokenizer, config)


### 3. 모델 및 트레이너 설정 ###
def load_tokenizer_and_model_for_train(config):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["general"]["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    tokenizer.add_special_tokens({"additional_special_tokens": config["tokenizer"]["special_tokens"]})
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def load_trainer_for_train(config, model, train_dataset, val_dataset):
    training_args = TrainingArguments(
        run_name=config["general"]["exp_name"],
        output_dir=config["general"]["output_dir"],
        overwrite_output_dir=config["training"]["overwrite_output_dir"],
        seed=config["training"]["seed"],
        report_to=config["training"]["report_to"],
        num_train_epochs=config["training"]["num_train_epochs"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        optim=config["training"]["optim"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        bf16=config["training"]["bf16"],
        fp16=config["training"]["fp16"],
        eval_strategy=config["training"]["eval_strategy"],
        save_strategy=config["training"]["save_strategy"],
        save_total_limit=config["training"]["save_total_limit"],
        logging_steps=config["training"]["logging_steps"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if config["training"]["do_eval"] else None,
    )
    return trainer


### 4. 모델 학습 ###
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = load_tokenizer_and_model_for_train(config)
    preprocessor = Preprocess(config["tokenizer"]["bos_token"], config["tokenizer"]["eos_token"])
    train_dataset, val_dataset = prepare_train_dataset(config, preprocessor, DATA_PATH, tokenizer)

    trainer = load_trainer_for_train(config, model, train_dataset, val_dataset)
    trainer.train()

    print(f"Best model path: {trainer.state.best_model_checkpoint}")
    return trainer.state.best_model_checkpoint

if __name__ == "__main__":
    main(config)
    wandb.finish()
