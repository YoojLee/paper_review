# Collaborative Filtering for Implicit Feedback Datasets

---

keyword: "personalized", "prior", "implicit feedback"

## abstract

recsys의 경우에는 사용자 선호를 모델링하기 위해서 purchase history나 watching habits browing activity 등을 추적함. (implicit feedback)
하지만 이런 implicit feedback 같은 경우에는 직접적으로 피드백을 설문하는 것(extensively researched explicit feedback)보다는 사용자들의 선호에 대한 직접적인 입력값이 없음. (좀 정보가 부족하다?)
특히, 뭘 "싫어하는지"에 대한 정보가 부족함.

### contributions

1. identify unique properties of implicit feedback datasets
2. treating the data as indication of positive and negative preference associated with vastly varying confidence levels.
3. leads to factor model which is especially tailored for implicit feedback recommenders
4. suggests a scalable optimization procedure (work in O(D) with data size D? i guess?) 그래서 얘가 지금 효율적이라는 건가?
5. performance가 어떤지?
    - successfully within a recommender system for tv shows (how successful? -> 어떤 식으로 evaluation할 건지가 애매한 듯)
    - compares favorably with well tuned implementations of other known methods (얼마나 favorable? 그걸 평가하는 기준이 뭔지? evaluation metrics를 잘 보자)
        - well tuned implementations와 비교하는데 그럼 이건 roughly tuned되었다는 건지? generality를 의미하는 건지?
    - give explanations to recommendations given by this factor model (explanable AI. factor model이 좀 explanable한가??)

# 1. Introduction

growing importance of recommender systems.
Previous recommender systems are based on profiling users and products (feature extracting), and *finding how to relate them*

Two different strategies of recommendations systems

1. content based approach
    - **creates a profile** for each user or product to characterize its nature
    - require gathering external information that might not be available or easy to collect (questionnaire 등 외부 데이터를 수집해야하기 때문에 데이터 구축비용이 큼)

2. Collabrative Filtering
    - relies only on past user behavior without requiring the creation of explicit profiles (데이터 구축 비용이 낮다. **explicit profiles가 무슨 의미인지는 아직 파악이 잘 안된다**)
    - analyzes relationships between users and interdependencies among products, in order to identify new user-item associations
    - domain free (content based approach는 domain free하지 않았나봄) + content based techniques 사용해서 profiling하기 어려운 데이터들도 다룰 수 있다는 점
    - 대략적으로 더 정확하지만 cold start 문제가 있음.
        - cold start: 추천시스템에 새로운 데이터가 들어올 경우 대응하기가 어려운 점. (content based approaches 같은 경우엔 이런 부분에 있어선 우위를 점하고 있음)

# 2. Preliminaries

# 3. Previous work

## 3.1. Neighborhood models

## 3.2. Latent factor models

# 4. Our model

# 5. Explaining recommendations

# 6. Experimental study

### Data description

### Evaluation methodology

### Evaluation results

# 7. Discussion
