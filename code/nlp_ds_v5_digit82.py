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
from rouge import Rouge
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback

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

config["general"]["output_dir"] = OUTPUT_PATH
config["general"]["data_path"] = DATA_PATH
config["tokenizer"]["bos_token"] = tokenizer.bos_token
config["tokenizer"]["eos_token"] = tokenizer.eos_token

exp_name = file_name.removeprefix("nlp_ds_")
now = datetime.now().strftime("%m%d-%H%M")
wandb.init(
    entity=TEAM,
    project="nlp-dialogue-summarization",
    name=f"{exp_name}_{now}",
    dir=os.path.join(OUTPUT_PATH, "logs"),
)
pprint(config)


### 2. 데이터 가공 및 데이터셋 클래스 ###
class Preprocess:
    def __init__(self, bos_token: str, eos_token: str) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.instruction = ""

    def make_set_as_df(self, file_path):
        return pd.read_csv(file_path)

    def clean_dialogue(self, text: str) -> str:
        if not isinstance(text, str): return text

        text = text.replace("\\n", "\n")
        text = re.sub(r"#\s*Person\s*([1-7])\s*#", r"#Person\1#", text)
        text = re.sub(r"(^|\n)(#Person\d#)(?!\:)", r"\1\2:", text)
        return text.strip()

    def make_input(self, dataset, is_test=False):
        encoder_inputs = []
        decoder_outputs = []

        for _, row in dataset.iterrows():
            dialogue = self.clean_dialogue(row["dialogue"])
            full_input = self.instruction + dialogue
            encoder_inputs.append(full_input)

            if not is_test:
                summary = row["summary"]
                summary = re.sub(r"(#Person\d#)\s+", r"\1", summary)
                decoder_outputs.append(summary + self.eos_token)

        if is_test:
            return encoder_inputs
        return encoder_inputs, decoder_outputs

class SummaryDataset(Dataset):
    def __init__(self, encoder_input, labels):
        self.encoder_input = encoder_input
        self.labels = labels

    def __getitem__(self, idx):
        return {
            "input_ids": self.encoder_input["input_ids"][idx],
            "attention_mask": self.encoder_input["attention_mask"][idx],
            "labels": self.labels["input_ids"][idx].clone(),
        }

    def __len__(self):
        return len(self.encoder_input["input_ids"])

class DatasetForInference(Dataset):
    def __init__(self, encoder_input, test_id):
        self.encoder_input = encoder_input
        self.test_id = test_id

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encoder_input.items()}
        item["ID"] = self.test_id[idx]
        return item

    def __len__(self):
        return len(self.test_id)

def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
    train_data = preprocessor.make_set_as_df(os.path.join(data_path, "train.csv"))
    val_data = preprocessor.make_set_as_df(os.path.join(data_path, "dev.csv"))

    encoder_train, decoder_out_train = preprocessor.make_input(train_data)
    encoder_val, decoder_out_val = preprocessor.make_input(val_data)

    def tokenize_fn(texts, max_len):
        return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)

    train_dataset = SummaryDataset(
        tokenize_fn(encoder_train, config["tokenizer"]["encoder_max_len"]),
        tokenize_fn(decoder_out_train, config["tokenizer"]["decoder_max_len"]),
    )
    val_dataset = SummaryDataset(
        tokenize_fn(encoder_val, config["tokenizer"]["encoder_max_len"]),
        tokenize_fn(decoder_out_val, config["tokenizer"]["decoder_max_len"]),
    )
    return train_dataset, val_dataset


### 3. trainer 및 핵심 로직 ###
def compute_metrics(tokenizer, pred):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids

    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p if p.strip() else "empty" for p in decoded_preds]

    results = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    return {key: value["f"] for key, value in results.items()}

def load_tokenizer_and_model(config, device, is_train=True):
    tokenizer = AutoTokenizer.from_pretrained(config["general"]["model_name"])
    tokenizer.add_special_tokens({"additional_special_tokens": config["tokenizer"]["special_tokens"]})

    if is_train:
        model = BartForConditionalGeneration.from_pretrained(config["general"]["model_name"], ignore_mismatched_sizes=True)
    else:
        model = BartForConditionalGeneration.from_pretrained(config["inference"]["ckt_path"])

    model.resize_token_embeddings(len(tokenizer))
    model.config.max_length = config["tokenizer"]["decoder_max_len"]
    return model.to(device), tokenizer


### 4. 모델 학습 ###
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate_model, tokenizer = load_tokenizer_and_model(config, device)

    preprocessor = Preprocess(config["tokenizer"]["bos_token"], config["tokenizer"]["eos_token"])
    train_ds, val_ds = prepare_train_dataset(config, preprocessor, config["general"]["data_path"], tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["general"]["output_dir"],
        overwrite_output_dir=config["training"]["overwrite_output_dir"],
        seed=config["training"]["seed"],
        report_to=config["training"]["report_to"],

        num_train_epochs=config["training"]["num_train_epochs"],
        learning_rate=config["training"]["learning_rate"],
        label_smoothing_factor=config["training"]["label_smoothing_factor"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        max_grad_norm=config["training"]["max_grad_norm"],
        optim=config["training"]["optim"],

        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        dataloader_num_workers=config["training"]["dataloader_num_workers"],
        group_by_length=config["training"]["group_by_length"],
        bf16=config["training"]["bf16"],

        eval_strategy=config["training"]["eval_strategy"],
        save_strategy=config["training"]["save_strategy"],
        save_total_limit=config["training"]["save_total_limit"],
        logging_strategy=config["training"]["logging_strategy"],
        logging_steps=config["training"]["logging_steps"],

        generation_max_length=config["training"]["generation_max_length"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        predict_with_generate=config["training"]["predict_with_generate"],
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=generate_model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = Seq2SeqTrainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(tokenizer, pred),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=config["training"]["early_stopping_patience"],
            early_stopping_threshold=config["training"]["early_stopping_threshold"],
        )],
    )

    trainer.train()
    wandb.finish()
    return trainer.state.best_model_checkpoint


### 5. 모델 추론 ###
def inference(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_tokenizer_and_model(config, device, is_train=False)
    preprocessor = Preprocess(config["tokenizer"]["bos_token"], config["tokenizer"]["eos_token"])

    test_df = pd.read_csv(os.path.join(config["general"]["data_path"], "test.csv"))
    encoder_test = preprocessor.make_input(test_df, is_test=True)

    tokenized_test = tokenizer(
        encoder_test,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config["tokenizer"]["encoder_max_len"],
    )
    test_ds = DatasetForInference(tokenized_test, test_df["fname"].tolist())
    dataloader = DataLoader(test_ds, batch_size=config["inference"]["batch_size"])

    summary = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            input_ids = item["input_ids"].to(device)
            attention_mask = item["attention_mask"].to(device)
            actual_input_len = attention_mask.sum(dim=1).max().item()
            dynamic_max_new_tokens = max(60, int(actual_input_len * 0.15))

            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config["inference"]["max_new_tokens"],
                min_length=config["inference"]["min_length"],
                num_beams=config["inference"]["num_beams"],
                repetition_penalty=config["inference"]["repetition_penalty"],
                length_penalty=config["inference"]["length_penalty"],
                no_repeat_ngram_size=config["inference"]["no_repeat_ngram_size"],
            )

            decoded_batch = tokenizer.batch_decode(output, skip_special_tokens=False)

            clean_decoded_batch = []
            for text in decoded_batch:
                sys_tokens = ["<s>", "</s>", "<pad>", "<unk>", "<mask>", "<usr>"]
                for token in sys_tokens:
                    text = text.replace(token, "")

                text = re.sub(r"(#Person\d#)\s+", r"\1", text)
                clean_decoded_batch.append(text.strip())

            summary.extend(clean_decoded_batch)

    submission_df = pd.DataFrame({"fname": test_df["fname"], "summary": summary})
    submission_df = submission_df.reset_index(drop=True)
    submission_df.to_csv(os.path.join(config["general"]["output_dir"], "submission.csv"), index=True, index_label="", header=True)

if __name__ == "__main__":
    best_ckt = main(config)
    if best_ckt:
        config["inference"]["ckt_path"] = best_ckt
        inference(config)
