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

def build_prompt(dialogue):
    ...
    # Note by Karyx💫: This part of the code is omitted to protect my intellectual property.


def get_topic(dialogue):
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model="solar-1-mini-chat",
                messages=build_prompt(dialogue),
                temperature=0,
                # top_p=0.1,
                seed=777,
            )
            raw_content = completion.choices[0].message.content.strip()

            if "[Topic]:" in raw_content:
                topic = raw_content.split("[Topic]:")[-1].strip()
            else:
                topic = raw_content.replace("[Topic]", "").strip()

            return topic.replace("\"", "").strip()

        except Exception as e:
            print(f"\n[에러] {e} - 재시도 중.. ({attempt+1}/3)")
            time.sleep(5)
    return "주제 파악 불가"

def run_inference(num_samples=10):
    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    target_df = test_df[:num_samples] if num_samples > 0 else test_df

    summaries = []
    start_time = time.time()

    for idx, row in tqdm(target_df.iterrows(), total=len(target_df)):
        summary = get_topic(row["dialogue"])
        summaries.append(summary)

        if (idx + 1) % 100 == 0:
            temp_df = pd.DataFrame({
                "fname": target_df["fname"][:idx+1],
                "topic": summaries
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
        "topic": summaries
    })

    save_name = "solar_test_final.csv" if num_samples == -1 else f"solar_test.csv"
    output.to_csv(os.path.join(OUTPUT_PATH, save_name), index=True, index_label="", header=True)

if __name__ == "__main__":
    run_inference(num_samples=-1)
