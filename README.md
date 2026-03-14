![banner_nlp](./assets/banner_nlp.jpg)

## **💻 Project Overview**
### Environment
- **OS:** Linux Ubuntu 20.04.6 LTS
- **GPU:** NVIDIA GeForce RTX 3090 (24GB)
- **NVIDIA Driver Version:** 535.86.10
- **CUDA Version:** 12.2 (Runtime: 12.1)
- **Tool:** VS Code (SSH)
- **Language:** Python 3.10.14
- **Prerequisites:** 한글 시각화를 위해 나눔 폰트 필요
```bash
apt update && apt install -y fonts-nanum
```

### Requirements
- accelerate==0.34.2
- bitsandbytes==0.45.0
- httpx==0.26.0
- ipykernel==7.2.0
- matplotlib==3.10.8
- openai==1.12.0
- pandas==2.3.3
- peft==0.12.0
- plotly==6.5.2
- python-dotenv==1.0.1
- rouge==1.0.1
- scikit-learn==1.7.2
- seaborn==0.13.2
- torch==2.4.1
- torchaudio==2.4.1
- torchvision==0.19.1
- transformers==4.46.1
- trl==0.12.1
- wandb==0.25.0
- wordcloud==1.9.6

---

## **📋 Competiton Info**
### DialogSum: A Real-life Scenario Dialogue Summarization (일상 대화 요약)
- 실제 일상생활(학교, 직장, 치료, 쇼핑, 여행 등)에서 가능한 다양한 시나리오 multi-turn 대화를 바탕으로 생성 요약문 작성
- 목표: 정확하고 일반화된 모델을 개발하여 요약문 생성
- 대화 스타일: 구어체 (최소 2명 ~ 최대 7명의 대화형식, 최소 2turn ~ 최대 60turn)
- 대화 도메인: 다양한 주제
- Senario: daily life

### 데이터셋 정보 (Dataset Info)
- 학습데이터: 12,457건
- 검증데이터: 499건
- 평가데이터: 499건
- 제출파일: 499건 (sample_submission.csv)
- 평가데이터는 학습데이터와 달리 dialogue 하나에 summary 3개 존재

### Feature 구성
- 학습데이터: fname (train_0부터), dialog, summary, topic
- 검증데이터: fname (dev_0부터), dialog, summary, topic
- 평가데이터: fname (test_0부터), dialog
- 제출파일: index (헤더명 없음, 0부터), fname, summary

### 정답 요약문 작성시 주요 기준
- 대화의 가장 중요한 정보를 전달
- 간략하게 (대화 길이의 20% 이내)
- 대화 내에서 중요한 명명된 개체를 보존 (사람 이름, 기업명 등)
- 관찰자의 관점에서 작성 (화자의 의도를 이해하고 작성)
- 은어나 약어 없이 공식적으로 사용되는 언어로 작성

### 규정 (Rule)
- DialogSum 데이터셋을 기반으로 한 모든 파생 데이터셋 및 파생 작업물 금지
- 무료로 사용 가능한 API에 한정하여 사용 가능 (Solar 모델은 사용 가능)

### 평가지표 (Evaluation Metric)
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

![equation](https://latex.codecogs.com/svg.image?\text{Score}=\frac{\sum_{i}^{N}\text{ROUGE-1-F1}(\text{pred},\text{gold}_i)}{N}&plus;\frac{\sum_{i}^{N}\text{ROUGE-2-F1}(\text{pred},\text{gold}_i)}{N}&plus;\frac{\sum_{i}^{N}\text{ROUGE-L-F1}(\text{pred},\text{gold}_i)}{N})

- Sentence Tokenization: 한국어 형태소 분석기를 통해 의미를 갖는 최소 단위인 형태소 단위로 문장을 쪼갠 뒤 모델이 생성한 문장과 정답 문장을 비교하여 ROUGE score 산출
- 3개의 summary에 대해서 개별적으로 점수를 산출한 뒤, 종합하여 최종 평가에 활용
- Metric 점수가 100점 만점이 아님 (3개의 정답 요약 문장 중 하나를 랜덤하게 선택하여 산출된 점수가 약 70점 정도)
- DialogSum 데이터셋은 Multi-Reference Dataset으로 multi-reference에 대한 average를 보는 것이 중요
- Public / Private은 대화 주제에 따라 50%씩 고르게 선정

---

## **⚙️ Components**
### Directory
```
├── archive/...                # legacy files (v1 ~ v5)
├── assets/...                 # README images & PDF
├── code/
│   ├── eda.ipynb              # EDA
│   ├── nlp_ds_v6_fail.py      # v6 (GroupKFold)
│   ├── nlp_ds_v6.py           # v6
│   ├── nlp_ds_v7_final.py     # v7 (final)
│   ├── solar_api_summary.py   # Solar API call (summary)
│   └── solar_api_topic.py     # Solar API call (topic)
├── config/                    # yaml file
│   ├── nlp_ds_v6.yaml         # 실행파일명과 동기화
│   └── nlp_ds_v7_final.yaml
├── data/                      # (이하 GitHub 관리안함)
│   ├── dev_solar.csv          # Solar API로 증강한 dev summary & topic
│   ├── dev.csv                # 검증데이터
│   ├── sample_submission.csv  # 제출파일 template
│   ├── test_solar.csv         # Solar API로 생성한 test topic
│   ├── test.csv               # 평가데이터
│   └── train.csv              # 학습데이터
├── experiments/               # (이하 GitHub 관리안함)
│   ├── checkpoint-####/...    # checkpoint directories
│   ├── logs/...               # WandB & logs
│   └── output.csv             # 추론 후 제출할 파일 생성
├── images/...                 # 시각화 images
├── .env                       # 경로설정
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## **💾 Data Descrption**
### EDA (Exploratory Data Analysis)
