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
OS_PATH = os.getenv('OS_PATH')
if not OS_PATH:
    raise ValueError('OS_PATH 환경변수가 설정되지 않았습니다!')

file_name = '_'.join(os.path.splitext(os.path.basename(__file__))[0].split('_')[:3])
EXP_PATH = os.path.join(OS_PATH, 'experiments', file_name)
CHECKPOINT_PATH = os.path.join(EXP_PATH, 'checkpoint-1000')
DATA_PATH = os.path.join(OS_PATH, 'data')

config_name = os.path.join(OS_PATH, 'config', f'{file_name}.yaml')
with open(config_name, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
tokenizer = AutoTokenizer.from_pretrained(config['general']['model_name'])

class Preprocess:
    def __init__(self, bos_token: str, eos_token: str) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token

    def clean_dialogue(self, text: str) -> str:
        text = text.replace('\\n', '\n')
        text = re.sub(r'(#Person\d+#)\s*[:\s]*', r'\1: ', text)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        return text

def post_process(dialogue, generated_text):
    mapping = re.findall(r'(#Person\d+#):\s*([가-힣a-zA-Z]+)', dialogue)
    result = generated_text

    for tag, name in mapping:
        if len(name) >= 2:
            result = result.replace(name, tag)

    result = re.sub(r'\s+', ' ', result).strip()
    return result

def final_cleaner(summary, dialogue):
    mapping = re.findall(r'(#Person\d+#):\s*([가-힣a-zA-Z\s]+)', dialogue)
    mapping.sort(key=lambda x: len(x[1].strip()), reverse=True)
    cleaned_summary = summary

    for tag, name in mapping:
        name = name.strip()
        if len(name) < 1: continue

        titles = [r'Mr\.', r'Mrs\.', r'Ms\.', r'Dr\.', 'Miss']
        for title in titles:
            pattern = rf'{title}\s*{name}'
            cleaned_summary = re.sub(pattern, tag, cleaned_summary, flags=re.IGNORECASE)

        cleaned_summary = cleaned_summary.replace(f'{name} 씨', tag)
        cleaned_summary = cleaned_summary.replace(f'{name} 님', tag)
        cleaned_summary = cleaned_summary.replace(name, tag)

    cleaned_summary = re.sub(r'(#Person\d+#)(\s*\1)+', r'\1', cleaned_summary)
    return cleaned_summary.strip()

def run_test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer.add_special_tokens({'additional_special_tokens': config['tokenizer']['special_tokens']})
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config['general']['model_name'],
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()
    torch.cuda.empty_cache()

    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    summaries = []
    stop_markers = [
        '### User:', '### 대화:', '###', 'Assistant:', 'User:',
        '</s>', '<pad>', '<|im_end|>', '</prompt>', '<usr>', '</reject>',
    ]

    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            preprocessor = Preprocess(tokenizer.bos_token, tokenizer.eos_token)
            dialogue = preprocessor.clean_dialogue(row['dialogue'])
            prompt = f'''### User:
다음 대화를 요약하세요.
**제약 사항:**
1. 반드시 한국어로 작성하세요.
2. 대화에 등장하는 실명(예: Jimmy, Mr. White 등)을 절대 사용하지 마세요.
3. 모든 인물은 반드시 대화에 표기된 대로 #Person1#, #Person2# 형태의 태그로만 지칭하세요.

### 예시 1 (기본):
대화: #Person1#: Jimmy, 내일 공원에서 보자. #Person2#: 알았어, 내일 봐.
요약: #Person1#은 #Person2#와 내일 공원에서 만나기로 약속합니다.

### 예시 2 (성+직함 처리):
대화: #Person1#: Dawson 씨, 이 서류 좀 봐주세요. #Person2#: 알겠네, Mr. White.
요약: #Person1#은 #Person2#에게 서류 검토를 요청하고, #Person2#는 이를 수락합니다.

### 예시 3 (제삼자 이름 보존):
대화: #Person1#: Brian이 자기 생일 파티에 오래. #Person2#: 정말? 우리 선물 사야겠다.
요약: #Person1#은 #Person2#에게 Brian의 생일 파티 소식을 전하고, 두 사람은 선물을 사기로 합니다.

### 대화:
{dialogue}

### Assistant:
'''

            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            input_len = inputs['input_ids'].shape[1]

            output_tokens = model.generate(
                **inputs,
                max_new_tokens=config['inference']['max_new_tokens'],
                min_length=config['inference']['min_length'],
                length_penalty=config['inference']['length_penalty'],
                num_beams=config['inference']['num_beams'],
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                repetition_penalty=config['inference']['repetition_penalty'],
                early_stopping=config['inference']['early_stopping'],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            torch.cuda.empty_cache()
            gen_text = tokenizer.decode(output_tokens[0][input_len:], skip_special_tokens=True).strip()

            stop_markers = stop_markers + ['<chosen>', '</prompt>', '<|im_start|>', '###', 'User:', 'Assistant:']
            for marker in stop_markers:
                if marker in gen_text:
                    gen_text = gen_text.split(marker)[0]

            gen_text = gen_text.strip()
            if gen_text and not gen_text.endswith(('.', '!', '?')):
                gen_text += '.'

            summaries.append(final_cleaner(gen_text, dialogue))

    test_df['summary'] = summaries
    output_path = os.path.join(EXP_PATH, 'output.csv')

    submission_df = test_df[['fname', 'summary']]
    submission_df = submission_df.reset_index(drop=True)
    submission_df.to_csv(output_path, index=True, index_label='', header=True)

if __name__ == '__main__':
    run_test()
