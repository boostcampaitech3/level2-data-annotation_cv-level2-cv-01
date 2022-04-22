## 1. 프로젝트 개요

📌 **프로젝트 개요**

**Goal** : 데이터를 구성하고 활용하는 방법에 집중하며 데이터의 중요성을 경험

***Problem Type*** 이미지에서 한글,영어 글자 검출 

***Metric.*** DetEval (F1Score)

***Data.***

학습데이터 : 총 536장의 이미지(*ICDAR*17 중 한글 데이터 샘플)

평가데이터 : 총 300장의 랜덤 크롤링된 이미지

**constraints :** model, loss, east_dataset, detect파일 수정금지, 오로지 데이터만 활용

**Base** : VGG16을 backbone으로한 Eastmodel으로 고정



📌 **개발환경 & 협업툴**

- **개발환경**

  | 개발환경       | 버전   |
  | -------------- | ------ |
  | VSCode         | 1.60.0 |
  | Albumentations | 1.1.0  |
  | GPU            | V100   |

- **협업 Tool** GitHub, Wandb, Notion 



## 2. 팀 구성 및 역할

📌 **팀 구성 및 역할**

- ***EDA → 강소망, 김기태***
- ***Engineering → 김창현***
- ***Augmentation → 김기태, 박기련***
- ***데이터 추가수집 → 박민수***
- ***Hyper parameter실험 → 공통*** 



## 3. 프로젝트 수행 절차 및 방법

![image-20220423000418006](https://raw.githubusercontent.com/variety82/imgForTypora/forUpload/img/image-20220423000418006.png)

### 1️⃣ *내부 평가 지표 설정*

📌 f1 score

- 하루 10회로 한정된 제출 기회를 효율적으로 사용하기 위해, 내부 평가 지표 설정
- 대회 기준 DetEval의 f1 score를 지표로 설정
- Baseline에서 출력이 되지 않아 코드 수정 



### 2️⃣ *Hyper Parameter Tuning*

📌 Hyper Parameter Tuning

- Modeling 부분 수정이 제한되므로, 최적의 Hyper Parameter Search 를 시도함
- Optimizer : SGD, ASGD, Adam, AdamW Scheduler : MultiStepLR, CosineAnnealing, CosineAnnealingWarmRestarts, StepLR
- 최종 Hyper Parameter : AdamW, CosineAnneling*



### *3️⃣ **Augmentation***

📌 **Albumentation**

1. **CLAHE**
1. **Emboss**
1. **RandomToneCurve**

**Default Augmentation**

1. Resize : 1024 x 1024
1. Rotate :  -10 ~ 10
1. Adjust Height : 0.8 ~ 1.2 (ratio)
1. Crop : 512 x 512

------

- Scene Text Recognition augmentation 적용, Border 부분에 augmentation 적용 시 효과가 있다는 논문참고
- 경계선부분을 명확하게 하는 효과들을 Group으로 하면서 가독성이 떨어지지 않게 적용 </aside>

### *4️⃣* *데이터*

📌 **데이터 추가**

- 기존 제공된 ICDAR17 536 개의 데이터는 매우 부족하다고 판단
- AISTAGE 측에서 제공한 약 1000개의 Annotaion 된 데이터를 추가 활용
- ICDAR19 데이터 10000 개 중, Korean 과 Latin 에 해당되는 데이터 2000 개를 추가 활용

------

**Result**

ICDAR17 : **0.4630**

ICDAR17 + AISTAGE : **0.5960 → 약 0.13 상승**

ICDAR17, 19 + AISTAGE : **0.6512 → 약 0.06 추가 상승**



📌 **데이터 검수 및 정제**

- AISTAGE 측에서 제공된 약 1000개의 데이터 검수
- [구글스프레드시트](https://docs.google.com/spreadsheets/d/1xYTSlKw1pQQ2m8Sd-CxegCDPbfaEjxH-99J6cXcLPXc/edit#gid=0)를 이용하여 기록
- 내부 특정 기준을 정하여 labeling이 불확실한 데이터 57개 제거

## 4. 프로젝트 수행 결과 및 분석

📌  **Result**

- **최종 제출** **선정 기준**

  - LB 의 F1 Score 상위 2개

- **선정 결과**

  1. ICDAR17, 19 + AISTAGE + Augmentation : **Public : 0.6512 / Private : 0.5815**
  1. ICDAR17, 19 + Augmentation : **Public : 0.6417 / Private : 0.5979**

- **최종 LB Score**

  **Public : 0.6417**      **Private : 0.5979**

📌 **분석**

- Public Score 가 **0.6066** 이었던 결과가 Private 에서 **0.6167** 로 가장 높았음

- Public Score 를 지표로 계속 학습을 하고 개선을 시도 했기 때문에,

  오히려 Public Data 에 Overfitting 이 발생 하였을 가능성이 있다 판단됨

- AISTAGE Dataset 의 경우, 추가하였을 때 점수 향상이 이루어지긴 하였지만 오히려 Private Score 결과가 떨어짐 → 잘못된 Annotation 정보의 악영향 



## Reference

____



- Data Augmentation for STR(Scene Text Recognition) - https://arxiv.org/pdf/2108.06949.pdf

  정리 - https://www.notion.so/dreamkkt/Augmentation-f066a7ce634944a08f6574bc9e8ecb79

- Border Augmentation 관련 - https://www.mdpi.com/2079-9292/9/1/117/htm

- ICDAR 17,19 Dataset - https://rrc.cvc.uab.es/?ch=8

- 깃헙 주소 - https://github.com/boostcampaitech3/level2-data-annotation_cv-level2-cv-01
