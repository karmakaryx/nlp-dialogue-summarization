import os
import re

import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

load_dotenv()
OS_PATH = os.getenv("OS_PATH")

file_name = "_".join(os.path.splitext(os.path.basename(__file__))[0].split("_")[:3])
OUTPUT_PATH = os.path.join(OS_PATH, "output", file_name)
CHECKPOINT_PATH = os.path.join(OUTPUT_PATH, "checkpoint-####")  # Best Checkpoint 입력!
DATA_PATH = os.path.join(OS_PATH, "data")

config_name = os.path.join(OS_PATH, "config", f"{file_name}.yaml")
with open(config_name, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained(config["general"]["model_name"])

def run_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer.add_special_tokens({"additional_special_tokens": config["tokenizer"]["special_tokens"]})
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config["general"]["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()

    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    summaries = []
    stop_markers = ["### User:", "### 대화:", "###", "Assistant:", "User:", "</s>", "<pad>", "<|im_end|>", "</prompt>"]

    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            dialogue = row["dialogue"].strip()
            prompt = f"""### System:
당신은 대화 요약 전문가입니다. 다음 [인물 매핑 규칙]과 [작성 규칙]을 엄격히 준수하세요.

[인물 매핑 규칙]
- **#Person1#**: 대화의 첫 번째 화자 (학습된 P1과 동일)
- **#Person2#**: 대화의 두 번째 화자 (학습된 P2와 동일)
- **#Person3#**: 대화의 세 번째 화자 (학습된 P3와 동일)

[작성 규칙]
1. **문장의 주어(시작점):** 반드시 #Person1#, #Person2# 등의 기호를 사용합니다.
2. **문장 속의 인물:** 주어가 아닌 다른 인물을 지칭할 때는 대화에 나온 실제 이름(예: Dave, Ann, 마이클)을 직접 사용하세요.
3. **금지 사항:** "그", "그녀", "그들", "두 사람", "다른 사람" 같은 대명사는 절대 사용하지 않습니다.
4. **어조:** "~합니다"로 끝나는 정중한 한국어 문장으로 작성하세요.

[예시]
대화:
#Person1#(P1): 이 가방 예쁘다.
#Person2#(P2): 생일 선물이야!
요약:
#Person2#는 #Person1#에게 생일 선물로 가방을 줍니다.

대화:
#Person1#(P1): Jane, 오늘 회의 어땠어?
#Person2#(P2): 좋았어요, #Person1#.
요약:
#Person1#은 #Person2#에게 회의가 어땠는지 묻고, #Person2#는 긍정적으로 답합니다.

### User:
다음 대화를 요약하세요.
대화:
{dialogue}

### Assistant:
"""

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            output_tokens = model.generate(
                **inputs,
                max_new_tokens=config["inference"].get("max_new_tokens", 200),
                min_length=config["inference"].get("min_length", 30),
                num_beams=config["inference"].get("num_beams", 2),
                no_repeat_ngram_size=config["inference"].get("no_repeat_ngram_size", 3),
                repetition_penalty=config["inference"].get("repetition_penalty", 1.2),
                early_stopping=config["inference"].get("early_stopping", True),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            gen_text = tokenizer.decode(output_tokens[0][input_len:], skip_special_tokens=True).strip()

            stop_markers = stop_markers + ["<chosen>", "</prompt>", "<|im_start|>", "###", "User:", "Assistant:"]
            for marker in stop_markers:
                gen_text = gen_text.split(marker)[0]

            gen_text = gen_text.replace("</s>", "").replace("<pad>", "").replace("<s>", "").replace("<|im_end|>", "").strip()
            gen_text = gen_text.replace("그들은", "#Person1#과 #Person2#는")
            gen_text = gen_text.replace("두 사람은", "#Person1#과 #Person2#는")

            gen_text = re.split(r"\. [A-Z]", gen_text)[0]
            if not gen_text.endswith("."):
                gen_text += "."

            summaries.append(gen_text)

    test_df["summary"] = summaries
    output_path = os.path.join(OUTPUT_PATH, "submission.csv")

    submission_df = test_df[["fname", "summary"]]
    submission_df.to_csv(output_path, index=False, header=False)

if __name__ == "__main__":
    run_test()
