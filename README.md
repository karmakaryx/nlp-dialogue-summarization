# **[Natural Language Processing] Dialogue Summarization**

## **💻 Project Overview**
### Environment
- **OS:** Linux Ubuntu 20.04.6 LTS
- **GPU:** NVIDIA GeForce RTX 3090
- **CUDA Version:** 12.2
- **Tool:** VS Code (SSH)
- **Language:** Python 3.11.14
- **Prerequisites:** 한글 시각화를 위해 나눔 폰트 필요
```bash
sudo apt update && sudo apt install -y fonts-nanum
```

### Requirements
- accelerate==0.34.2
- ipykernel==7.2.0
- matplotlib==3.10.8
- pandas==2.2.2
- plotly==6.5.2
- python-dotenv==1.0.1
- rouge==1.0.1
- seaborn==0.13.2
- torch==2.4.1
- torchaudio==2.4.1
- torchvision==0.19.1
- transformers==4.44.2
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
- 간략하게(대화 길이의 20% 이내)
- 대화 내에서 중요한 명명된 개체를 보존 (사람 이름, 기업명 등)
- 관찰자의 관점에서 작성 (화자의 의도를 이해하고 작성)
- 은어나 약어 없이 공식적으로 사용되는 언어로 작성

### 규정 (Rule)
- DialogSum 데이터셋을 기반으로 한 모든 파생 데이터셋 및 파생 작업물 금지
- 무료로 사용 가능한 API에 한정하여 사용 가능

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
├── assets/...                 # README images
├── code/
│   ├── eda.ipynb              # EDA
│   ├── nlp_ds_v1_baseline.py  # baseline code
│   ├── nlp_ds_v1_yaml.py      # config 분리
│   ├── nlp_ds_v2_eda.py       # EDA 분리
│   └── nlp_ds_v2_model.py     # 새 모델
├── config/                    # yaml file
│   ├── nlp_ds_v1_yaml.yaml    # 실행파일명과 동기화
│   ├── ...
│   └── nlp_ds_v2_model.yaml
├── data/                      # (이하 GitHub 관리안함)
│   ├── dev.csv                # 검증데이터
│   ├── sample_submission.csv  # 제출파일 template
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
#### 1. 데이터의 구조적 무결성 검증
> 훈련데이터의 마지막 인덱스 번호(0-12459)와 안내된 건수(12457)가 불일치해 fname 누락 여부 검사<br>
> 분석 방법: fname에서 숫자 추출하여 0부터 최댓값까지의 집합과 실제 데이터 집합 간의 차집합 연산 수행<br>
> 적용 범위: 훈련, 검증, 평가 데이터 및 제출 파일에 모두 수행<br>
> 결과: 숨겨진 결측치에 의해 인덱스 불연속성 확인 (훈련 3건 [10933, 10972, 11473], 검증 1건 [475], 평가 1건 [466])

> 중복 대화 확인: 0건

#### 2. Qualitative Glimpse
> 비정형 데이터인 일상 대화이나 채팅 대화와 달리 약어나 이모지 없이 formal style을 가짐<br>

> 화자 수: #Person1# 부터 #Person7# 까지<br>
> 각각의 발화자를 구분하기 위해 #Person”N”#: 을 사용하며, 발화자의 대화가 끝나면 \n 으로 구분

> 개인정보 마스킹 처리: 전화번호(#PhoneNumber#), 주소(#Address#), 생년월일(#DateOfBirth#), 여권번호(#PassportNumber#), 사회보장번호(#SSN#), 신용카드번호(#CardNumber#), 차량번호(#CarNumber#), 이메일주소(#Email#)

#### 3. Topic Inspection
> 전체 12,457건 대화 중 토픽이 (공백 정제 후에도) 9,235종에 달하는 심각한 분산 현상<br>
> Treemap 결과, 전체 토픽 종류 (9,235종) 대비 약 87% (8,041종), 전체 데이터 대비 약 64.5%가 1회성 토픽에 해당
> Top 5: 음식 주문 (130), 취업 면접 (109), 길 안내 (66), 호텔 체크인 (40), 아파트 임대 (30)

![treemap](./images/treemap.jpg)

> 토픽 빈도수 분포로 시각화한 결과, 전형적인 long-tail 형태를 넘어 log를 적용해야 꼬리라도 보일 것 같다..🫥
![longtail](./images/longtail.png)

#### 4. 문자열 길이
> 토픽이 너무 많아 문자열 길이도 제대로 볼 수가 없다! (겹쳐서 시꺼먼게 전부 무한 토픽들..)
![violin_plot](./images/violin_plot.png)

> 10건 이하 토픽을 그룹화하니 겨우 상황 파악 가능<br>
> 평균 406자, 최소 84자, 최대 2,165자로 입력 데이터의 최대 길이가 model의 max length 크게 초과 (모델 교체 검토)<br>
> 근데 또 10건 이하 토픽이 문자열 긴 놈도 유난히 많아요. 이상치 점이 선이 되고 있다..환장하것다..
![box_plot](./images/box_plot.png)

#### 5. Dialogue Inspection
> 토픽별로 단어 빈도를 대략적으로 확인하기 위해 최다 토픽 5건에 대해 Word Clouds 시각화<br>
> 일반적이거나 의미없는 단어들은 간단한 불용어사전을 작성해 필터링하니 주제별로 키워드가 확실히 보인다.<br>
> (예약했어요 손님 방이 인상적이다.. 아-파트아파트아-파트 🎶)

![topic1](./images/wordcloud_01.png)
| Topic 2 | Topic 3 |
| :---: | :---: |
| ![topic2](./images/wordcloud_02.png) | ![topic3](./images/wordcloud_03.png) |
| **Topic 4** | **Topic 5** |
| ![topic4](./images/wordcloud_04.png) | ![topic5](./images/wordcloud_05.png) |

> 대화당 등장인원 통계를 내보니 최대 참여인원은 7명인데 7명까진 훼이크고 대부분 둘이 주고받는 대화다..<br>
> 점점 통계 내는게 의미가 없는 기분이다. 😂<br>
> 그러나 화자 태그 포맷 규칙, 개행 규칙이 위반된 대화 건(eda.ipynb 참조)들이 있어 data cleaning 필요
![participants](./images/participants.png)

### Data Preprocessing
- test.csv와 submission의 index를 일치시키기 위해 left join 병합으로 dataframe mapping을 시도했으나, 이후 제출 파일 또한 평가 데이터와 동일한 인덱스가 누락됨을 발견, 만일을 위해 assert만 수행
- special_tokens에 화자를 #Person7#까지 모두 추가하고 마스킹된 개인정보 태그도 모두 추가
- data cleaning 관련

---

## **🔍 Hypothesis Testing**

---

## **🧠 Modeling**
### Model Description

### Modeling Process

---

## **📊 Experiment Logger**
<table>
  <thead>
    <tr>
      <th align="center">NO.</th>
      <th align="center">DATE</th>
      <th align="center">MODEL</th>
      <th align="center">KEY CHANGES</th>
      <th align="center">R1</th>
      <th align="center">R2</th>
      <th align="center">RL</th>
      <th align="center" colspan="2">SCORE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">05</td>
      <td align="center">260227</td>
      <td>KoBART(digit82)</td>
      <td>비정상 토큰 이슈 해결</td>
      <td align="center">0.5127</td>
      <td align="center">0.3229</td>
      <td align="center">0.4157</td>
      <td align="center">41.7098</td>
      <td align="center">S</td>
    </tr>
    <tr>
      <td align="center">04</td>
      <td align="center">260226</td>
      <td>KoBART(digit82)</td>
      <td>비정상 토큰 이슈 디버깅</td>
      <td align="center">0.3824</td>
      <td align="center">0.1746</td>
      <td align="center">0.3056</td>
      <td align="center">28.7529</td>
      <td align="center">F</td>
    </tr>
    <tr>
      <td align="center">03</td>
      <td align="center">260226</td>
      <td>KoBART(digit82)</td>
      <td>config 분리</td>
      <td align="center">0.2420</td>
      <td align="center">0.1469</td>
      <td align="center">0.1921</td>
      <td align="center">19.3655</td>
      <td align="center">F</td>
    </tr>
    <tr>
      <td align="center">02</td>
      <td align="center">260226</td>
      <td>KoBART(digit82)</td>
      <td>refactoring</td>
      <td align="center">0.5691</td>
      <td align="center">0.3760</td>
      <td align="center">0.4808</td>
      <td align="center">47.5295</td>
      <td align="center">S</td>
    </tr>
    <tr>
      <td align="center">01</td>
      <td align="center">260226</td>
      <td>KoBART(digit82)</td>
      <td>baseline code</td>
      <td align="center">0.5676</td>
      <td align="center">0.3737</td>
      <td align="center">0.4807</td>
      <td align="center">47.4018</td>
      <td align="center">S</td>
    </tr>
  </tbody>
</table>
![wandb_01](./assets/wandb_01.png)

---

## **💡 Insights from Trial and Error**
#### [#03. ROUGE 19.3655] nlp_ds_v1_yaml.py
- **증상:** 화자 태그(#Person#)가 &lt;unused68&gt; 같은 비정상 토큰과 깨진 한자(㗡)로 도배되어 있음
- **원인:** clean_up_tokenization_spaces=True 설정으로 인한 decoding sequence 왜곡 및 한자 생성
- **조치:** 해당 옵션 제거 및 정규표현식을 통한 비정상 토큰 후처리 로직 도입
- **교훈:** 한국어 특수 토큰 추가시 tokenizer의 자동 공백 정리 기능을 지양해야 함

---

## **🚀 Result**
### Champion Model Info

### Leaderboard Rank: No. 1 🏆 ()

---

## **📜 Version Log**
### V1: digit82/kobart-summarization
> **nlp_ds_v1_baseline.py:**
- Jupyter Notebook을 Python script로 변환하며 발생하는 warnings & runtime errors 해결
- code formatting: PEP 8 적용
- code refactoring: 중복코드 제거 등
- 하드웨어 사양에 라이브러리 최적화
- 환경 설정: 데이터, 출력, 로그 경로 등

> **nlp_ds_v1_yaml.py:**
- config 설정값 .yaml 파일로 관리
- 학습데이터 기준으로 화자 수, 개인정보 마스킹 yaml에 추가
- 실험명, 로그명, 환경파일명 등을 파일명, UTC와 동기화하여 자동화
- WandB로 checkpoint upload 중지
- hyperparameter 수정: batch size, gradient steps
- 데이터 로딩 방식 변경: on-the-fly tokenization
- tokenizer 공백 자동 정리 적용
- downgrade library versions

### V2: EDA
> **nlp_ds_v2_eda.py:**
- 본격 EDA를 위해 Jupyter Notebook 파일로 분리
- tokenizer 공백 자동 정리 로직 제거 & 정규표현식 후처리
- model 중복 호출 제거
- WandB 로그 범위 확대

---

## **🛠️ etc.**
### Reference
- [[GitHub] DialogSum: A Real-life Scenario Dialogue Summarization Dataset](https://github.com/cylnlp/dialogsum)
- [[arXiv] DialogSum: A Real-Life Scenario Dialogue Summarization Dataset (Chen et al., ACL 2021)](https://arxiv.org/abs/2105.06762)
- [[Kaggle] DialogSum Corpus: A Large-Scale Dataset for Dialogue Summarization and Topic Gen](https://www.kaggle.com/datasets/marawanxmamdouh/dialogsum/data)
- Solar API

### 프로젝트 회고
지난 CV 대회 때는 리더보드 점수 올리기에만 매몰되어 중도에 실험기록을 WandB에만 맡기는 바람에 산출물 작성시에 (정신도 몽롱한 상태에서) 애로사항이 좀 있었습니다.<br>
따라서 이번 대회는 실험별, 버전별, 파일별로 어떤 변화가 있었는지 최대한 상세하게 기록하려고 노력했습니다.<br>
여행에서 남는건 사진 뿐이고 코드로 말해야 할 엔지니어한테 남는건 README와 PDF 뿐인가 싶어 기분이 좀 그렇습니다만😑 기록은 그때그때 해놓는게 정신건강에 좋은 것 같습니다..<br>

<br>
