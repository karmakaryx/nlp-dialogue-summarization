import pandas as pd
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from rouge import Rouge

load_dotenv()
client = OpenAI(
    api_key=os.getenv("SOLAR_API"),
    base_url=os.getenv("SOLAR_URL"),
)

OS_PATH = os.getenv("OS_PATH")
DATA_PATH = os.path.join(OS_PATH, "data")
OUTPUT_PATH = os.path.join(OS_PATH, "output")

rouge = Rouge()
def compute_metrics(pred, gold):
    results = rouge.get_scores(pred, gold, avg=True)
    result = {key: value["f"] for key, value in results.items()}
    return result

def get_len_limit(dialogue):
    char_len = len(dialogue)
    ...
    # Note by Karyx💫: This part of the code is omitted to protect my intellectual property.

def build_prompt(dialogue):
    limit_text = get_len_limit(dialogue)
    ...
    # Note by Karyx💫: This part of the code is omitted to protect my intellectual property.


def summarization(dialogue):
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model="solar-1-mini-chat",
                messages=build_prompt(dialogue),
                temperature=0,
                # top_p=0.2,
                seed=777,
            )
            raw_content = completion.choices[0].message.content.strip()

            if "[최종 요약]:" in raw_content:
                summary = raw_content.split("[최종 요약]:")[-1].strip()
            else:
                summary = raw_content.split("\n")[-1].strip()

            return summary.replace("\"", "").strip()

        except Exception as e:
            print(f"\n[에러 발생] {e} - 5초 후 다시 시도합니다.. ({attempt+1}/3)")
            time.sleep(5)
    return "요약 실패"

def run_validation(num_samples=10):
    val_df = pd.read_csv(os.path.join(DATA_PATH, "dev.csv"))
    target_df = val_df[:num_samples] if num_samples > 0 else val_df

    preds = []
    golds = []

    for idx, row in tqdm(target_df.iterrows(), total=len(target_df)):
        pred_summary = summarization(row["dialogue"])
        preds.append(pred_summary)
        golds.append(row["summary"])

    all_scores = []
    for p, g in zip(preds, golds):
        try:
            score = compute_metrics(p, g)
            all_scores.append(score)
        except:
            continue

    avg_scores = {
        "rouge-1": sum(s["rouge-1"] for s in all_scores) / len(all_scores),
        "rouge-2": sum(s["rouge-2"] for s in all_scores) / len(all_scores),
        "rouge-l": sum(s["rouge-l"] for s in all_scores) / len(all_scores),
    }

    print("\n" + "="*50)
    print("검증 결과 (Validation Score)")
    print(f"ROUGE-1: {avg_scores['rouge-1']:.4f}")
    print(f"ROUGE-2: {avg_scores['rouge-2']:.4f}")
    print(f"ROUGE-L: {avg_scores['rouge-l']:.4f}")
    print("="*50)

    output = pd.DataFrame({
        "fname": target_df["fname"],
        "pred_summary": preds,
        "gold_summary": golds
    })
    output.to_csv(os.path.join(OUTPUT_PATH, "solar_dev.csv"), index=False)

def run_inference(num_samples=10):
    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    target_df = test_df[:num_samples] if num_samples > 0 else test_df

    summaries = []
    start_time = time.time()

    for idx, row in tqdm(target_df.iterrows(), total=len(target_df)):
        summary = summarization(row["dialogue"])
        summaries.append(summary)

        # 중간 저장
        if (idx + 1) % 100 == 0:
            temp_df = pd.DataFrame({
                "fname": target_df["fname"][:idx+1],
                "summary": summaries
            })
            temp_df.to_csv(os.path.join(OUTPUT_PATH, f"solar_temp_{idx+1}.csv"), index=False)

            elapsed = time.time() - start_time
            if elapsed < 60:
                wait = 60 - elapsed + 2
                print(f"\n[안전] {wait:.1f}초 동안 잠시 쉽니다..")
                time.sleep(wait)
            start_time = time.time()

    output = pd.DataFrame({
        "fname": target_df["fname"],
        "summary": summaries
    })

    save_name = "solar_test_final.csv" if num_samples == -1 else f"solar_test.csv"
    output.to_csv(os.path.join(OUTPUT_PATH, save_name), index=True, index_label="", header=True)

if __name__ == "__main__":
    # run_validation(num_samples=10)
    run_inference(num_samples=-1)
