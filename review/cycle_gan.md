# Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
---
## 1. Intorduction
- image-to-image translation: **이미지 쌍을 활용해** 입력 이미지와 출력 이미지 간의 mapping을 학습
- 하지만, 이미지 쌍 데이터는 그 비용이 매우 크다.
- 저자는 논문에서 unpaired image-to-image translation 방법을 제안함.
  - 하나의 이미지 collection에서 특징을 잡아내고 다른 이미지 collection으로 해당 특징을 어떻게 녹여낼 수 있을지 찾아냄. (w.o any paired training examples)
- 