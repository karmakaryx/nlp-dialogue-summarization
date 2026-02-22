# **[Natural Language Processing] Dialogue Summarization**

## **💻 Project Overview**
### Environment
- **OS:** Linux Ubuntu 20.04.6 LTS
- **GPU:** NVIDIA GeForce RTX 3090
- **CUDA Version:** 12.2
- **Tool:** VS Code (SSH)
- **Language:** Python 3.11.14

### Requirements

---

## **📋 Competiton Info**
### DialogSum: A Real-life Scenario Dialogue Summarization
- 실제 일상생활에서 가능한 다양한 시나리오 multi-turn 대화를 바탕으로 생성 요약문 작성
- 대화 스타일: 구어체 (대화형식)
- 대화 도메인: 다양한 주제
- Senario: daily life

### 데이터셋 정보 (Dataset Info)
- 학습데이터: 12457
- 검증데이터: 499
- 평가데이터: 499
- 평가데이터는 학습데이터와 달리 dialogue 하나에 summary 3개가 존재

### 정답 요약문 작성시 주요 기준
- 대화의 가장 중요한 정보를 전달
- 간략하게(대화 길이의 20% 이내)
- 대화 내에서 중요한 명명된 개체를 보존 (사람 이름, 기업명 등)
- 관찰자의 관점에서 작성 (화자의 의도를 이해하고 작성)
- 은어나 약어 없이 공식적으로 사용되는 언어로 작성

### 평가지표 (Evaluation Metric)
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

![equation](https://latex.codecogs.com/svg.image?\text{Score}=\frac{\sum_{i}^{N}\text{ROUGE-1-F1}(\text{pred},\text{gold}_i)}{N}&plus;\frac{\sum_{i}^{N}\text{ROUGE-2-F1}(\text{pred},\text{gold}_i)}{N}&plus;\frac{\sum_{i}^{N}\text{ROUGE-L-F1}(\text{pred},\text{gold}_i)}{N})

- 3개의 summary에 대해서 개별적으로 점수를 산출한 뒤, 종합하여 최종 평가에 활용
- Public / Private은 대화 주제에 따라 50%씩 고르게 선정

---
