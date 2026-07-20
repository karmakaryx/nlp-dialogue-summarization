### 1. 데이터 및 환경설정 ###
import os
import random
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from dotenv import load_dotenv
from rouge import Rouge
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback

load_dotenv()
TEAM = os.getenv("TEAM")
OS_PATH = os.getenv("OS_PATH")
DATA_PATH = os.path.join(OS_PATH, "data")
OUTPUT_PATH = os.path.join(OS_PATH, "output")

file_name = "_".join(os.path.splitext(os.path.basename(__file__))[0].split("_")[:3])
config_name = os.path.join(OS_PATH, "config", f"{file_name}.yaml")
with open(config_name, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


### 2. 데이터 가공 및 데이터셋 클래스 구축 ###
def clean_text(text: str) -> str:
    if not isinstance(text, str): return text

    text = text.replace("\\n", "\n")
    text = re.sub(r"#\s*Person\s*([1-7])\s*#", r"#Person\1#", text)
    text = re.sub(r"(?<!^)(#Person[1-7]#:)", r"\n\1", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

class Preprocess:
    def __init__(self, bos_token: str, eos_token: str) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token

    @staticmethod
    def make_set_as_df(file_path, is_train=True):
        df = pd.read_csv(file_path)

        if "dialogue" in df.columns:
            df["dialogue"] = df["dialogue"].apply(clean_text)
        if is_train and "summary" in df.columns:
            df["summary"] = df["summary"].apply(clean_text)

        if "topic" in df.columns:
            df["topic"] = df["topic"].apply(clean_text)

        cols = ["fname", "dialogue"]
        if "topic" in df.columns: cols.append("topic")
        if is_train: cols.append("summary")
        return df[cols]

    def make_input(self, dataset, is_test=False):
        if "topic" in dataset.columns:
            encoder_input = dataset.apply(
                lambda x: f"[TOPIC] {x.get('topic', '일반 대화')} [DIALOGUE] {x['dialogue']}",
                axis=1,
            )
        else:
            encoder_input = dataset["dialogue"].apply(lambda x: f"[DIALOGUE] {x}")

        if is_test:
            return encoder_input.tolist()
        else:
            decoder_input = dataset["summary"].apply(lambda x: self.bos_token + str(x))
            decoder_output = dataset["summary"].apply(lambda x: str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()

class SummaryDataset(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, pad_token_id):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.pad_token_id = pad_token_id

    def __getitem__(self, idx):
        label_tensor = self.labels["input_ids"][idx].clone()
        label_tensor[label_tensor == self.pad_token_id] = -100

        return {
            "input_ids": self.encoder_input["input_ids"][idx],
            "attention_mask": self.encoder_input["attention_mask"][idx],
            "decoder_input_ids": self.decoder_input["input_ids"][idx],
            "decoder_attention_mask": self.decoder_input["attention_mask"][idx],
            "labels": label_tensor,
        }

    def __len__(self):
        return len(self.encoder_input["input_ids"])

class DatasetForInference(Dataset):
    def __init__(self, encoder_input, test_id, len):
        self.encoder_input = encoder_input
        self.test_id = test_id
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encoder_input.items()}
        item["ID"] = self.test_id[idx]
        return item

    def __len__(self):
        return self.len

D_THRESHOLDS = [70, 91, 110, 125, 146, 160, 185, 219, 255, 295, 930]

def get_dynamic_config(input_len):
    if input_len <= D_THRESHOLDS[0]:   return {"max_length": 19, "min_length": 0}
    elif input_len <= D_THRESHOLDS[1]: return {"max_length": 22, "min_length": 0}
    elif input_len <= D_THRESHOLDS[2]: return {"max_length": 25, "min_length": 0}
    elif input_len <= D_THRESHOLDS[3]: return {"max_length": 30, "min_length": 0}
    elif input_len <= D_THRESHOLDS[4]: return {"max_length": 35, "min_length": 0}

    elif input_len <= D_THRESHOLDS[5]: return {"max_length": 40, "min_length": 0}
    elif input_len <= D_THRESHOLDS[6]: return {"max_length": 48, "min_length": 0}
    elif input_len <= D_THRESHOLDS[7]: return {"max_length": 56, "min_length": 0}

    elif input_len <= D_THRESHOLDS[8]: return {"max_length": 65, "min_length": 0}
    elif input_len <= D_THRESHOLDS[9]: return {"max_length": 75, "min_length": 0}

    else: return {"max_length": 158, "min_length": 0}

def mbr_ensemble(candidates):
    ...
    # Note by Karyx💫: This part of the code is omitted to protect my intellectual property.

def prepare_fold_dataset(config, preprocessor, train_df, val_df, tokenizer):
    encoder_input_train, decoder_input_train, decoder_output_train = preprocessor.make_input(train_df)
    encoder_input_val, decoder_input_val, decoder_output_val = preprocessor.make_input(val_df)

    def tokenize_fn(inputs, max_len):
        return tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            return_token_type_ids=False,
        )

    train_inputs_dataset = SummaryDataset(
        tokenize_fn(encoder_input_train, config["tokenizer"]["encoder_max_len"]),
        tokenize_fn(decoder_input_train, config["tokenizer"]["decoder_max_len"]),
        tokenize_fn(decoder_output_train, config["tokenizer"]["decoder_max_len"]),
        tokenizer.pad_token_id,
    )

    val_inputs_dataset = SummaryDataset(
        tokenize_fn(encoder_input_val, config["tokenizer"]["encoder_max_len"]),
        tokenize_fn(decoder_input_val, config["tokenizer"]["decoder_max_len"]),
        tokenize_fn(decoder_output_val, config["tokenizer"]["decoder_max_len"]),
        tokenizer.pad_token_id,
    )

    return train_inputs_dataset, val_inputs_dataset


### 3. trainer 및 training args 구축 ###
def compute_metrics(tokenizer, pred):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=False)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

    def cleanup_tokens(text):
        for token in [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token, "<usr>"]:
            if token:
                text = text.replace(token, "")

        text = re.sub(r"(#Person\d#)\s+", r"\1", text).strip()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    cleaned_preds = [cleanup_tokens(p) for p in decoded_preds]
    cleaned_labels = [cleanup_tokens(l) for l in decoded_labels]

    cleaned_preds = [p if len(p) > 0 else "empty" for p in cleaned_preds]

    results = rouge.get_scores(cleaned_preds, cleaned_labels, avg=True)
    return {key: value["f"] for key, value in results.items()}

def load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset, fold_dir):
    training_args = Seq2SeqTrainingArguments(
        output_dir=fold_dir,
        overwrite_output_dir=config["training"]["overwrite_output_dir"],
        seed=config["training"]["seed"],
        report_to=config["training"]["report_to"],
        num_train_epochs=config["training"]["num_train_epochs"],
        learning_rate=float(config["training"]["learning_rate"]),
        label_smoothing_factor=config["training"]["label_smoothing_factor"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        optim=config["training"]["optim"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        dataloader_num_workers=config["training"]["dataloader_num_workers"],
        fp16=config["training"]["fp16"],
        eval_strategy=config["training"]["eval_strategy"],
        save_strategy=config["training"]["save_strategy"],
        save_total_limit=config["training"]["save_total_limit"],
        logging_strategy=config["training"]["logging_strategy"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        greater_is_better=config["training"]["greater_is_better"],
        generation_max_length=config["training"]["generation_max_length"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        predict_with_generate=config["training"]["predict_with_generate"],
    )

    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=config["training"]["early_stopping_patience"],
        early_stopping_threshold=config["training"]["early_stopping_threshold"],
    )

    return Seq2SeqTrainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics=lambda pred: compute_metrics(tokenizer, pred),
        callbacks=[MyCallback],
    )


### 4. 모델 학습 (K-Fold) ###
def main(config):
    seed_everything(config["training"]["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config["general"]["model_name"])
    tokenizer.add_special_tokens({"additional_special_tokens": config["tokenizer"]["special_tokens"]})
    preprocessor = Preprocess(tokenizer.bos_token, tokenizer.eos_token)

    train_df = preprocessor.make_set_as_df(os.path.join(DATA_PATH, "train.csv"), is_train=True)
    val_df = preprocessor.make_set_as_df(os.path.join(DATA_PATH, "dev.csv"), is_train=True)
    total_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    kf = KFold(n_splits=config["training"]["n_splits"], shuffle=True, random_state=config["training"]["seed"])

    exp_name = file_name.removeprefix("nlp_ds_")
    now = datetime.now().strftime("%m%d-%H%M")
    best_model_paths = []

    for fold, (t_idx, v_idx) in enumerate(kf.split(total_df)):
        print(f"Fold {fold+1} Training...")

        generate_model = BartForConditionalGeneration.from_pretrained(config["general"]["model_name"], ignore_mismatched_sizes=True)
        generate_model.resize_token_embeddings(len(tokenizer))
        generate_model.to(device)

        wandb.init(
            entity=TEAM,
            project="nlp-dialogue-summarization",
            name=f"{exp_name}_{now}_fold{fold+1}",
            dir=os.path.join(OUTPUT_PATH, "logs"),
            config=config,
            reinit=True,
        )

        curr_train = total_df.iloc[t_idx]
        curr_val = total_df.iloc[v_idx]

        train_ds, val_ds = prepare_fold_dataset(config, preprocessor, curr_train, curr_val, tokenizer)

        fold_dir = os.path.join(OUTPUT_PATH, f"fold_{fold+1}")
        trainer = load_trainer_for_train(config, generate_model, tokenizer, train_ds, val_ds, fold_dir)
        trainer.train()
        best_model_paths.append(trainer.state.best_model_checkpoint)

        del generate_model
        del trainer
        torch.cuda.empty_cache()
        wandb.finish()

    return best_model_paths, tokenizer


### 5. 모델 추론 ###
def inference(config, ckt_paths, tokenizer):
    seed_everything(config["training"]["seed"])
    torch.use_deterministic_algorithms(False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preprocessor = Preprocess(tokenizer.bos_token, tokenizer.eos_token)

    test_file_path = os.path.join(DATA_PATH, "test.csv")
    test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
    test_topic_path = os.path.join(DATA_PATH, "test_solar.csv")
    if os.path.exists(test_topic_path):
        topic_df = pd.read_csv(test_topic_path)
        test_data = pd.merge(test_data, topic_df[["fname", "topic"]], on="fname", how="left")
        test_data["topic"] = test_data["topic"].fillna("일반 대화")
    else:
        print("Warning: Test topic file not found. Proceeding without topics.")

    test_id = test_data["fname"]
    encoder_input_test = preprocessor.make_input(test_data, is_test=True)
    test_tokenized = tokenizer(
        encoder_input_test,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config["tokenizer"]["encoder_max_len"],
        return_token_type_ids=False,
    )

    test_dataset = DatasetForInference(test_tokenized, test_id, len(encoder_input_test))
    dataloader = DataLoader(
        test_dataset,
        batch_size=config["inference"]["batch_size"],
        shuffle=config["inference"]["shuffle"],
        worker_init_fn=seed_worker,
        num_workers=config["inference"]["num_workers"],
    )

    folds_preds = []

    for fold_idx, path in enumerate(ckt_paths):
        print(f"\n[Fold {fold_idx+1}] Inferencing with: {path}")

        model = BartForConditionalGeneration.from_pretrained(path, ignore_mismatched_sizes=True)
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
        model.eval()

        fold_summary = []
        with torch.no_grad():
            for item in tqdm(dataloader, desc=f"Fold {fold_idx+1}"):
                input_ids = item["input_ids"].to(device)
                attention_mask = item["attention_mask"].to(device)

                avg_input_len = attention_mask.sum(dim=1).float().mean().item()
                d_cfg = get_dynamic_config(avg_input_len)

                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=d_cfg["max_length"],
                    min_length=d_cfg["min_length"],
                    length_penalty=config["inference"]["length_penalty"],
                    num_beams=config["inference"]["num_beams"],
                    no_repeat_ngram_size=config["inference"]["no_repeat_ngram_size"],
                    repetition_penalty=config["inference"]["repetition_penalty"],
                    forced_eos_token_id=tokenizer.eos_token_id,
                    early_stopping=config["inference"]["early_stopping"],
                )

                for ids in generated_ids:
                    result = tokenizer.decode(ids, skip_special_tokens=False)

                    for token in [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token, "[TOPIC]", "[DIALOGUE]", "<usr>"]:
                        if token: result = result.replace(token, "")

                    titles = ["Mr.", "Ms.", "Mrs.", "Sr.", "Dr.", "Prof.", "Inc.", "Co.", "Ltd.", "St.", "Rd."]
                    last_punc = max(result.rfind("."), result.rfind("!"), result.rfind("?"))
                    if last_punc != -1:
                        words = result[:last_punc + 1].split()
                        if words and words[-1] in titles:
                            result = result[:result.rfind(words[-1])].strip()
                        else:
                            result = result[:last_punc + 1].strip()

                    result = re.sub(r"(#Person\d#)\s+", r"\1", result).strip()
                    result = re.sub(r"\s+", " ", result).strip()

                    if result and not result.endswith((".", "!", "?")):
                        result += "."

                    fold_summary.append(result)

        folds_preds.append(fold_summary)
        del model
        torch.cuda.empty_cache()

    # MBR ensemble
    final_summary = []
    num_samples = len(test_data)

    for i in tqdm(range(num_samples), desc="MBR Selection"):
        sample_candidates = [fold[i] for fold in folds_preds]
        voted_result = mbr_ensemble(sample_candidates)
        final_summary.append(voted_result)

    output_df = pd.DataFrame({"fname": test_data["fname"], "summary": final_summary})
    output_df = output_df.reset_index(drop=True)
    output_path = os.path.join(OUTPUT_PATH, "output_ensemble.csv")
    output_df.to_csv(output_path, index=True, index_label="", header=True)

if __name__ == "__main__":
    best_paths, trained_tokenizer = main(config)
    if best_paths:
        inference(config, best_paths, trained_tokenizer)

# if __name__ == "__main__":
#     seed_everything(config["training"]["seed"])
#     tokenizer = AutoTokenizer.from_pretrained(config["general"]["model_name"])
#     tokenizer.add_special_tokens({"additional_special_tokens": config["tokenizer"]["special_tokens"]})

#     best_paths = [
#         os.path.join(OUTPUT_PATH, "fold_1", "checkpoint-3952"),
#         os.path.join(OUTPUT_PATH, "fold_2", "checkpoint-4160"),
#         os.path.join(OUTPUT_PATH, "fold_3", "checkpoint-3328"),
#         os.path.join(OUTPUT_PATH, "fold_4", "checkpoint-3328"),
#         os.path.join(OUTPUT_PATH, "fold_5", "checkpoint-2912"),
#     ]

#     inference(config, best_paths, tokenizer)
