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
from transformers import AutoTokenizer, BartForConditionalGeneration, EarlyStoppingCallback
from transformers import GenerationConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments

load_dotenv()
TEAM = os.getenv("TEAM")
OS_PATH = os.getenv("OS_PATH")

file_name = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_PATH = os.path.join(OS_PATH, "output", file_name)

config_name = os.path.join(OS_PATH, "config", f"{file_name}.yaml")  # 이 파일과 연결된 yaml은 nlp_ds_v2.yaml을 수정해서 사용할 것
if os.path.exists(config_name):
    with open(config_name, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
else:
    raise FileNotFoundError(f"{config_name} 파일이 존재하지 않습니다!")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained(config["general"]["model_name"])

exp_name = file_name.removeprefix("nlp_ds_")
now = datetime.now().strftime("%m%d-%H%M")

config["general"]["data_path"] = os.path.join(OS_PATH, "data")
config["general"]["output_dir"] = OUTPUT_PATH
config["general"]["exp_name"] = f"{exp_name}_{now}"

config["training"]["logging_dir"] = os.path.join(OUTPUT_PATH, "logs")
config["wandb"]["entity"] = TEAM
config["inference"]["result_path"] = OUTPUT_PATH

config["tokenizer"]["bos_token"] = tokenizer.bos_token
config["tokenizer"]["eos_token"] = tokenizer.eos_token
config["inference"]["remove_tokens"] = [
    "<usr>",
    tokenizer.bos_token,
    tokenizer.eos_token,
    tokenizer.pad_token,
]

wandb.init(
    entity=config["wandb"]["entity"],
    project=config["wandb"]["project"],
    name=config["general"]["exp_name"],
    dir=config["training"]["logging_dir"],
)

pprint(config)
print("*" * 10, config["general"]["exp_name"], "*" * 10)

data_path = config["general"]["data_path"]
train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
val_df = pd.read_csv(os.path.join(data_path, "dev.csv"))


### 2. 데이터 가공 및 데이터셋 클래스 구축 ###
class Preprocess:
    def __init__(self, bos_token: str, eos_token: str) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token

    @staticmethod
    def make_set_as_df(file_path, is_train=True):
        if is_train:
            df = pd.read_csv(file_path)
            train_df = df[["fname", "dialogue", "summary"]]
            return train_df
        else:
            df = pd.read_csv(file_path)
            test_df = df[["fname", "dialogue"]]
            return test_df

    def make_input(self, dataset, is_test=False):
        if is_test:
            encoder_input = dataset["dialogue"]
            return encoder_input.tolist()
        else:
            encoder_input = dataset["dialogue"]
            decoder_input = dataset["summary"].apply(lambda x: self.bos_token + str(x))
            decoder_output = dataset["summary"].apply(lambda x: str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()

class SummaryDataset(Dataset):
    def __init__(self, df, tokenizer, config):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        inputs = self.tokenizer(
            item["dialogue"],
            max_length=self.config["tokenizer"]["encoder_max_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            item["summary"],
            max_length=self.config["tokenizer"]["decoder_max_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
        }

    def __len__(self):
        return len(self.df)

class DatasetForInference(Dataset):
    def __init__(self, encoder_input, test_id, len):
        self.encoder_input = encoder_input
        self.test_id = test_id.tolist()
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item["ID"] = self.test_id[idx]
        return item

    def __len__(self):
        return self.len

def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
    train_file_path = os.path.join(data_path, "train.csv")
    val_file_path = os.path.join(data_path, "dev.csv")

    train_df = preprocessor.make_set_as_df(train_file_path)
    val_df = preprocessor.make_set_as_df(val_file_path)

    train_inputs_dataset = SummaryDataset(train_df, tokenizer, config)
    val_inputs_dataset = SummaryDataset(val_df, tokenizer, config)

    return train_inputs_dataset, val_inputs_dataset


### 3. trainer 및 training args 구축 ###
def compute_metrics(config, tokenizer, pred):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids

    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
    decoded_labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)

    replaced_predictions = decoded_preds.copy()
    replaced_labels = decoded_labels.copy()
    remove_tokens = config["inference"]["remove_tokens"]
    for token in remove_tokens:
        replaced_predictions = [sentence.replace(token, " ") for sentence in replaced_predictions]
        replaced_labels = [sentence.replace(token, " ") for sentence in replaced_labels]

    for i in range(min(3, len(replaced_predictions))):
        print(f"PRED {i}: {replaced_predictions[i]}")
        print(f"GOLD {i}: {replaced_labels[i]}\n")

    results = rouge.get_scores(replaced_predictions, replaced_labels, avg=True)
    result = {key: value["f"] for key, value in results.items()}
    return result

def load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset):
    print("-" * 10, "Make training arguments", "-" * 10)
    training_args = Seq2SeqTrainingArguments(
        output_dir=config["general"]["output_dir"],
        run_name=config["general"]["exp_name"],
        overwrite_output_dir=config["training"]["overwrite_output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        learning_rate=config["training"]["learning_rate"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        optim=config["training"]["optim"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        dataloader_num_workers=config["training"]["dataloader_num_workers"],
        eval_strategy=config["training"]["eval_strategy"],
        save_strategy=config["training"]["save_strategy"],
        save_total_limit=config["training"]["save_total_limit"],
        fp16=config["training"]["fp16"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        seed=config["training"]["seed"],
        logging_dir=config["training"]["logging_dir"],
        logging_strategy=config["training"]["logging_strategy"],
        predict_with_generate=config["training"]["predict_with_generate"],
        generation_max_length=config["training"]["generation_max_length"],
        do_train=config["training"]["do_train"],
        do_eval=config["training"]["do_eval"],
        report_to=config["training"]["report_to"],
        disable_tqdm=config["training"]["disable_tqdm"],
    )

    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=config["training"]["early_stopping_patience"],
        early_stopping_threshold=config["training"]["early_stopping_threshold"],
    )
    print("-" * 10, "Make training arguments complete", "-" * 10)

    print("-" * 10, "Make trainer", "-" * 10)
    trainer = Seq2SeqTrainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred),
        callbacks = [MyCallback],
    )
    print("-" * 10, "Make trainer complete", "-" * 10)
    return trainer

def load_tokenizer_and_model_for_train(config, device):
    print("-" * 10, "Load tokenizer & model", "-" * 10)
    special_tokens_dict = {"additional_special_tokens": config["tokenizer"]["special_tokens"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    generate_model = BartForConditionalGeneration.from_pretrained(
        config["general"]["model_name"],
        ignore_mismatched_sizes=True,
    )

    generate_model.resize_token_embeddings(len(tokenizer))

    gen_config = GenerationConfig.from_model_config(generate_model.config)
    generate_model.generation_config = gen_config

    generate_model.to(device)
    print(generate_model.config)
    print("-" * 10, "Load tokenizer & model complete", "-" * 10)

    return generate_model, tokenizer


### 4. 모델 학습 ###
def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"torch version: {torch.__version__}")

    generate_model, tokenizer = load_tokenizer_and_model_for_train(config, device)
    print(f"special tokens: {tokenizer.special_tokens_map}")

    preprocessor = Preprocess(config["tokenizer"]["bos_token"], config["tokenizer"]["eos_token"])
    data_path = config["general"]["data_path"]
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config, preprocessor, data_path, tokenizer)

    trainer = load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset)
    trainer.train()

    return trainer.state.best_model_checkpoint


### 5. 모델 추론 ###
def prepare_test_dataset(config, preprocessor, tokenizer):
    test_file_path = os.path.join(config["general"]["data_path"], "test.csv")
    test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
    test_id = test_data["fname"]
    print(f"test_data:\n{test_data['dialogue'][0]}")

    encoder_input_test = preprocessor.make_input(test_data, is_test=True)

    test_tokenized_encoder_inputs = tokenizer(
        encoder_input_test, return_tensors="pt", padding=True,
        add_special_tokens=True, truncation=True, max_length=config["tokenizer"]["encoder_max_len"],
        return_token_type_ids=False,
    )

    test_encoder_inputs_dataset = DatasetForInference(test_tokenized_encoder_inputs, test_id, len(encoder_input_test))
    return test_data, test_encoder_inputs_dataset

def load_tokenizer_and_model_for_test(config, device):
    print("-" * 10, "Load tokenizer & model", "-" * 10)
    ckt_path = config["inference"]["ckt_path"]
    special_tokens_dict = {"additional_special_tokens": config["tokenizer"]["special_tokens"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    generate_model = BartForConditionalGeneration.from_pretrained(
        ckt_path,
        ignore_mismatched_sizes=True,
    )

    generate_model.config.forced_eos_token_id = None
    gen_config = GenerationConfig.from_model_config(generate_model.config)
    gen_config.forced_eos_token_id = None
    generate_model.generation_config = gen_config

    generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)
    print("-" * 10, "Load tokenizer & model complete", "-" * 10)

    return generate_model, tokenizer

def inference(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(torch.__version__)

    generate_model , tokenizer = load_tokenizer_and_model_for_test(config,device)
    preprocessor = Preprocess(config["tokenizer"]["bos_token"], config["tokenizer"]["eos_token"])

    test_data, test_encoder_inputs_dataset = prepare_test_dataset(config, preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=config["inference"]["batch_size"])

    summary = []
    text_ids = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            text_ids.extend(item["ID"])
            generated_ids = generate_model.generate(
                input_ids=item["input_ids"].to(device),
                no_repeat_ngram_size=config["inference"]["no_repeat_ngram_size"],
                early_stopping=config["inference"]["early_stopping"],
                max_length=config["inference"]["generate_max_length"],
                num_beams=config["inference"]["num_beams"],
            )

            for ids in generated_ids:
                result = tokenizer.decode(ids, skip_special_tokens=False)
                result = result.split("\n")[0].strip()
                result = re.sub(r"<[^>]+>", "", result)  # 시스템 토큰(<...>) 삭제
                result = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s.,?!#]", "", result)  # 한글, 숫자, 영문, 공백, 마침표, # 외의 문자는 다 삭제
                summary.append(result.strip())

    remove_tokens = config["inference"]["remove_tokens"]
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token, " ") for sentence in preprocessed_summary]

    output = pd.DataFrame({
        "fname": test_data["fname"],
        "summary": preprocessed_summary,
    })

    result_path = config["inference"]["result_path"]
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    submission_df = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
    if not output["fname"].equals(submission_df["fname"]):
        raise ValueError("추론 결과의 fname과 제출 양식의 fname이 일치하지 않습니다!")

    output.to_csv(os.path.join(result_path, "output.csv"), index=False)
    return output

if __name__ == "__main__":
    best_path = main(config)

    if best_path:
        config["inference"]["ckt_path"] = best_path
        output = inference(config)
        print(output.head())
    else:
        print("체크포인트가 없습니다!")

    wandb.finish()
