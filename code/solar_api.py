import pandas as pd
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from rouge import Rouge

load_dotenv()
client = OpenAI(
    api_key=os.getenv('SOLAR_API'),
    base_url=os.getenv('SOLAR_URL'),
)

OS_PATH = os.getenv('OS_PATH')
DATA_PATH = os.path.join(OS_PATH, 'data')
EXP_PATH = os.path.join(OS_PATH, 'experiments')

rouge = Rouge()
def compute_metrics(pred, gold):
    results = rouge.get_scores(pred, gold, avg=True)
    result = {key: value['f'] for key, value in results.items()}
    return result

def get_len_limit(dialogue):
    char_len = len(dialogue)

    if char_len < 200:
        return "반드시 **전체 길이 50자 이내 한 문장**으로 매우 짧게 핵심만 요약하세요."
    elif char_len > 550:
        return "반드시 **전체 길이 170자 이내 세 문장**으로 구체적인 수치와 고유명사를 포함하여 요약하세요."
    else:
        return "반드시 **전체 길이 90자 이내** 두 문장 이내로 핵심 토픽 위주로 요약하세요."

def build_prompt(dialogue):
    limit_text = get_len_limit(dialogue)

    system_instructions = (
        "당신은 불필요한 수식어와 문장 연결을 제거하는 **'한국어 대화 압축 요약 전문가'**입니다.\n"
        f"1. {limit_text}\n"
        "2. 반드시 한국어로만 일목요연하게 요약하세요.\n"
        "3. 약 10~20턴 이상의 긴 대화라도 핵심 정보 위주로 1번의 전체 길이 제한을 지켜서 작성하세요.\n"
        "4. 말투는 격식 있는 존댓말('~합니다', '~했습니다')을 사용하세요.\n"
        "5. 다소 건조하고 딱딱한 번역체를 사용하세요.\n"
        "6. #Person1#, #Person2#, #Person3# 식별자를 유지하세요.\n"
        "7. 관찰자의 관점에서 화자의 의도를 이해하고 **일반화하여** 요약하세요.\n"
        "8. 구구절절 구체적인 사항을 설명하지 말고 **대략적으로** 요약하세요.\n"
        "9. 부사나 형용사를 최소화하고 명사, 동사 어간 위주로 구성하세요.\n"
        "10. 가장 중요한 주제를 대표할 수 없는 일반 명사를 나열하지 마세요.\n"
        "11. 사람 이름, 기업명 등 대화 내에서 중요한 고유 명칭은 보존하세요."
    )

    turns = len([line for line in dialogue.splitlines() if line.strip()])

    # Few-shot 예시 (Assistant 역할 분리 및 CoT 적용)
    example_dialogue_1 = (
        "#Person1#: 나 진짜 건강에 안 좋은 음식 좀 그만 먹어야겠어.\n"
        "#Person2#: 맞아, 무슨 말인지 알아. 나도 요즘 건강하게 먹으려고 하거든.\n"
        "#Person1#: 요즘은 뭐 먹어?\n"
        "#Person2#: 주로 과일이랑 채소, 닭고기 먹지.\n"
        "#Person1#: 그래, 그게 훨씬 건강해 보이네."
    )
    example_assistant_1 = (
        "[핵심 사건]: #Person1#의 식습관 개선 결심과 #Person2#의 건강 식단 공유.\n"
        "[최종 요약]: #Person1#은 건강에 나쁜 음식을 끊기로 결심하고, #Person2#는 자신의 건강 식단을 공유합니다."
    )

    # example_dialogue_2 = (
    #     "#Person1#: 알겠어요. 근데 이거 검정색인데, 저 검정색 신발은 별로 안 좋아해요. 너무 칙칙하거든요.\n"
    #     "#Person2#: 음, 그래도 검정색이 분홍색보다는 낫잖아. 분홍색은 여자애들 신는 느낌이잖아.\n"
    #     "#Person1#: 그러면 왜 엄마는 검정색 신발을 신고 있어요?\n"
    #     "#Person2#: 어... 알았어. 네가 이겼어. 어서 계산하고 가자.\n"
    #     "#Person1#: 와, 고마워요, 엄마."
    # )
    # example_assistant_2 = (
    #     "[핵심 사건]: 신발 색상을 두고 벌인 모자간의 실랑이와 #Person1#의 설득 성공.\n"
    #     "[최종 요약]: #Person1#은 어머니를 설득하여 원하는 신발을 구매합니다."
    # )

    # example_dialogue_3 = (
    #     "#Person1#: 안녕하세요. 어떤 도움을 드릴까요?\n"
    #     "#Person2#: 안녕하세요, 저는 고정자산 대출에 대해 상담하러 왔습니다.\n"
    #     "#Person1#: 네, 물론입니다. 혹시 저희와 기본 계좌가 있으신가요?\n"
    #     "#Person2#: 네, 기본 계좌도 있고 대출 증명서도 있습니다.\n"
    #     "#Person1#: 아, 그럼 잘됐네요. 저희는 신용 등급과 상환 능력을 기준으로 최종 결정을 내리게 됩니다.\n"
    #     "#Person2#: 음, 신용 등급은 문제 없습니다. 그 점은 확실히 말씀드릴 수 있어요."
    # )
    # example_assistant_3 = (
    #     "[핵심 사건]: #Person2#의 고정자산 대출 상담과 대출 심사 절차 및 기준 안내.\n"
    #     "[최종 요약]: #Person2#는 고정자산 대출을 상담하기 위해 방문했습니다. #Person1#은 #Person2#의 계좌 보유 여부와 대출 증명서를 확인한 뒤, 신용 등급과 상환 능력을 기준으로 최종 결정을 내린다고 설명합니다."
    # )

    # example_dialogue_4 = (
    #     "#Person1#: 아비가일, 너희 결혼식은 어땠어?\n"
    #     "#Person2#: 우리 남편이랑 친구 두 명만 증인으로 초대해서 구청에서 결혼식을 했어. 그렇지만 세 번 파티를 했지.\n"
    #     "#Person1#: 세 번이나? 그것도 꽤 많네. 돈 많이 들었겠다!\n"
    #     "#Person2#: 뭐, 우리 남편이랑 나는 서로 다른 나라 출신이고, 지금은 또 다른 나라에 살고 있어서 각 나라에서 한 번씩 하기로 했어. 사실 그렇게 비싸진 않았어.\n"
    #     "#Person1#: 부모님이 결혼식 못 본 거 서운해하시진 않았어?\n"
    #     "#Person2#: 부모님이 오셨으면 좋았겠지만, 비행기 타고 오실 비용이 여의치 않아서 우리도 갈 수 없었고, 부모님도 이해하셨어.\n"
    #     "#Person1#: 남편 쪽 가족은 너희 가족이랑 만날 기회가 있었어?\n"
    #     "#Person2#: 우리 고향에서 결혼식을 할 때 남편 부모님이 비행기 타고 오셔서 우리 가족과 만날 수 있었어. 다른 사람들은 결혼식에 엄청난 돈을 쓰지만, 우리 둘은 간단하게 하기로 했어.\n"
    #     "#Person1#: 그거 좋네. 신혼여행은 갔어?\n"
    #     "#Person2#: 1주년을 기념하여 아프리카로 신혼여행을 갔어.\n"
    #     "#Person1#: 정말 전통적인 결혼식은 아니었구나.\n"
    #     "#Person2#: 전혀. 하지만 우리 결혼 생활도 전통적이지 않아서 우리한테 딱 맞았어!"
    # )
    # example_assistant_4 = (
    #     "[핵심 사건]: 아비가일의 국가별 세 차례 파티를 포함한 비전통적 결혼 방식과 이에 대한 만족감.\n"
    #     "[최종 요약]: #Person1#이 아비가일에게 결혼식에 대해 묻습니다. 아비가일과 남편은 서로 다른 나라 출신으로 제3국에서 결혼하여 세 번의 파티를 열었습니다. 부모님과 가족이 일부 결혼식에 참석하지 못했지만, 이해했습니다. 신혼여행은 1주년을 기념하여 아프리카로 갔습니다. 아비가일은 비전통적인 결혼식이 비전통적인 결혼 생활과 완벽하게 어우러진다고 느낍니다."
    # )

    # 메시지 리스트 구성
    return [
        {"role": "system", "content": system_instructions},

        # Few-shot 예시
        {"role": "user", "content": f"### [데이터 정보: 총 5턴의 대화]\n{example_dialogue_1}\n\n요약 과정 및 결과:"},
        {"role": "assistant", "content": example_assistant_1},

        # {"role": "user", "content": f"### [데이터 정보: 총 5턴의 대화]\n{example_dialogue_2}\n\n요약 과정 및 결과:"},
        # {"role": "assistant", "content": example_assistant_2},

        # {"role": "user", "content": f"### [데이터 정보: 총 6턴의 대화]\n{example_dialogue_3}\n\n요약 과정 및 결과:"},
        # {"role": "assistant", "content": example_assistant_3},

        # {"role": "user", "content": f"### [데이터 정보: 총 12턴의 대화]\n{example_dialogue_4}\n\n요약 과정 및 결과:"},
        # {"role": "assistant", "content": example_assistant_4},

        # 실제 입력 데이터
        {"role": "user", "content": f"### [데이터 정보: 총 {turns}턴의 대화]\n{dialogue}\n\n요약 과정 및 결과:"},
    ]


def summarization(dialogue):
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model='solar-1-mini-chat',
                messages=build_prompt(dialogue),
                temperature=0,
                top_p=0.2,
                seed=777,
            )
            raw_content = completion.choices[0].message.content.strip()

            if "[최종 요약]:" in raw_content:
                summary = raw_content.split("[최종 요약]:")[-1].strip()
            else:
                summary = raw_content.split("\n")[-1].strip()

            return summary.replace('"', '').strip()

        except Exception as e:
            print(f"\n[에러 발생] {e} - 5초 후 다시 시도합니다... ({attempt+1}/3)")
            time.sleep(5)
    return "요약 실패"

def run_validation(num_samples=10):
    val_df = pd.read_csv(os.path.join(DATA_PATH, 'dev.csv'))
    target_df = val_df[:num_samples] if num_samples > 0 else val_df
    target_df = val_df[200 : 200 + num_samples]

    preds = []
    golds = []

    for idx, row in tqdm(target_df.iterrows(), total=len(target_df)):
        pred_summary = summarization(row['dialogue'])
        preds.append(pred_summary)
        golds.append(row['summary'])

    all_scores = []
    for p, g in zip(preds, golds):
        try:
            score = compute_metrics(p, g)
            all_scores.append(score)
        except:
            continue

    avg_scores = {
        'rouge-1': sum(s['rouge-1'] for s in all_scores) / len(all_scores),
        'rouge-2': sum(s['rouge-2'] for s in all_scores) / len(all_scores),
        'rouge-l': sum(s['rouge-l'] for s in all_scores) / len(all_scores),
    }

    print('\n' + '='*50)
    print('검증 결과 (Validation Score)')
    print(f"ROUGE-1: {avg_scores['rouge-1']:.4f}")
    print(f"ROUGE-2: {avg_scores['rouge-2']:.4f}")
    print(f"ROUGE-L: {avg_scores['rouge-l']:.4f}")
    print('='*50)

    output = pd.DataFrame({
        'fname': target_df['fname'],
        # 'dialogue': target_df['dialogue'],
        'pred_summary': preds,
        'gold_summary': golds
    })
    output.to_csv(os.path.join(EXP_PATH, 'solar_dev.csv'), index=False)

def run_inference(num_samples=10):
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    target_df = test_df[:num_samples] if num_samples > 0 else test_df

    summaries = []
    start_time = time.time()

    for idx, row in tqdm(target_df.iterrows(), total=len(target_df)):
        summary = summarization(row['dialogue'])
        summaries.append(summary)

        # 중간 저장
        if (idx + 1) % 100 == 0:
            temp_df = pd.DataFrame({
                'fname': target_df['fname'][:idx+1],
                'summary': summaries
            })
            temp_df.to_csv(os.path.join(EXP_PATH, f'solar_temp_{idx+1}.csv'), index=False)

            elapsed = time.time() - start_time
            if elapsed < 60:
                wait = 60 - elapsed + 2
                print(f'\n[안전] {wait:.1f}초 동안 잠시 쉽니다...')
                time.sleep(wait)
            start_time = time.time()

    output = pd.DataFrame({
        'fname': target_df['fname'],
        'summary': summaries
    })

    save_name = 'solar_test_final.csv' if num_samples == -1 else f'solar_test.csv'
    output.to_csv(os.path.join(EXP_PATH, save_name), index=True, index_label='', header=True)

if __name__ == '__main__':
    # run_validation(num_samples=10)
    run_inference(num_samples=-1)
