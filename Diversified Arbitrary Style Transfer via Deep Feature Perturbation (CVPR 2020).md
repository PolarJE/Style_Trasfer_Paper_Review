# Diversified Arbitrary Style Transfer via Deep Feature Perturbation (CVPR 2020)
(Style Transfer 결과의 다양성을 높이기 위한 논문. Gram Matrix 및 WCT를 잘 알아야하며 꽤나 수학적 지식이 있어야 이해할 수 있음)

## 0. Abstract
- style transfer는 일반화, 다양성, 확장성이 제한됨. 
- 따라서 본 논문에서는 이러한 한계를 해결하고 **다양한 임의의 스타일 전송** 을 위한 간단하면서도 효과적인 방법을 제안
- 우리 방법의 핵심 아이디어는 **deep feature perturbation(DFP)** 로 직교 랜덤 노이즈 행렬(orthogonal random noise matrix)를 사용하여 원본 스타일 정보를 변경하지 않고 deep image feature를 변경(perturb)함

## 1. Introduction
<세 가지 주요한 한계점>
1. 제한된 일반화 : 한번 훈련되면 feed-forward network는 특정한 스타일에 연결되어 다른 스타일로 일반화할 수 없다.
2. 제한된 다양성 : 한정된 데이터셋의 미니배치의 변동에 패널티를 적용하여 다양성을 학습하고 다양성 손실의 가중치를 작은 값으로 설정해야하므로 다양성의 정도가 제한됨.
3. 확장성 부족 : 접근 방식을 다른 메서드로 확장하려면 훈련 전략 및 네트워크 구조에 대해 다루기 힘든 수정이 필요함. 이는 일부 learning-based methods에는 유용하나 최근의 learning-free methods에는 적합하지 않음. 
- 우리가 사용할 중요한 insight는 style representation으로 널리 사용되는 Gram matrix가 무한한 수의 다른 feature maps에 대응할 수 있고, 이러한 feature maps에서 재구성된 이미지는 우리가 기대하는 다양한 결과라는 것임.
- 이제 다양성의 문제는 동일한 Gram matrix로 다른 feature map을 얻는 방법의 문제로 전환됨.
- Gram matrix를 분해하고 whitening  and coloring transforms (WCTs)으로 그들의 매칭을 분리하는 Li et al. [19]의 연구에서 영감을 받아, 우리는 deep feature perturbation (DFP)를 제안
- 다양성은 orthogonal noise matrix를 사용하여 원래 스타일 정보를 변경하지 않고 DCNN에서 추출한 image feature map을 perturb(변경,교란)함으로써 얻어짐


그램 행렬을 분해하고 화이트닝 및 컬러 변환(WCT)으로 이들의 일치를 분리하는 Li의 연구에서 영감을 받아, 우리는 다양화된 임의 스타일 전송을 달성하기 위해 **심층 특징 섭동(DFP)** 을 제안한다. 우리의 다양성은 직교 노이즈 매트릭스를 사용하여 원래 스타일 정보를 변경하지 않고 DCNN에서 추출한 이미지 기능 맵을 교란함으로써 얻어진다. 즉, 교란된 기능 맵은 서로 다르지만 모두 동일한 Gram 행렬을 가지고 있다. 이해하기 쉽도록 Gram 매트릭스를 스타일 표현으로 간주하고 동일한 Gram 매트릭스를 가진 서로 다른 피쳐 맵이 동일한 스타일별 피쳐 공간을 공유한다고 정의한다.

#### 본 논문의 주요한 contributions
- 다양한 arbitrary style transfer를 위해 original style information은 바꾸지 않으면서 orthogonal noise matrix(직교 노이즈 매트릭스)에 의해 deep image feature maps를 perturbing(섭동)하는 deep feature perturbation(DFP)를 사용할 것을 제안한다.
- 다양한 style transfer tasks에 사용되는 기존 WCT기반 기법에 쉽게 통합될 수 있다.

## 2. Related Work
**WCT-based Mthods (참고 : https://jeinalog.tistory.com/23)**
- Whitening and Coloring Transforms (WCT) : 공분산까지 맞추는 모델
  - 공분산까지 맞춘다는 것은, 좀 더 Fine tuning이 되도록 하는 것
  - Auto Encoder는 똑같이 학습시킨 후, Whitening & Coloring을 진행
  - Whitening이란? 이미지의 색을 쫙 빼는 작업, content를 유지하면서 style을 빼는 작업으로 볼 수 있음
  - Coloring이란? 이미지에 색을 다시 입히는 작업, Style을 입히는 것으로 볼 수 있음

**Diversified Methods.**
- Li et al.[18] : feed-forward networks가 다양한 outputs을 생성할 수 있도록 diversity loss를 도입
- Ulyanov et al.[30] : Julesz ensemble을 샘플링한 generative networks를 훈련할 수 있는 새로운 공식을 제안

**제안하는 방법은 WCT[19]를 기반으로 하며 WCT 기반 방법에 쉽게 통합되어 다양한 결과를 생성할 수 있음. 이전에는 모든 스타일에 대해 독립적으로 네트워크를 훈련시켰어야 했는데 제안하는 방법은 learning-free하며 다양한 임의 스타일 전송을 할 수 있다.**

## 3. Style-Specific Feature Space
이미지의 스타일을 정의하는 것은 매우 까다로운 문제이며 아직 통합된 결론에 도달 못함.
<img src="/img/Diversified Arbitrary Style Transfer_1.PNG"></img><br/>

- (비공식적으로) 스타일은 색상, 붓놀림(brush strokes), line drawing 등과 같은 visual attributes의 집합으로 간주됨
- 최근에 Gatys et al. [7, 6, 8]은 예술적 이미지에 대해 새로운 스타일 표현을 제안 → Gram matrix
- 동일한 S에 속하는 Features는 style characteristics에서 perceptually 동일함
  <img src="/img/Diversified Arbitrary Style Transfer_2.PNG"></img><br/>
- 본 논문에서 제안하는 방법으로 얻은 다양한 perturbed feature maps의 Gram matrices는 완전히 동일할 수 있다.

## 4. Deep Feature Perturbation(DFP)
#### Deep Feature Perturbation(DFP) = Li et al. [19] + whitening  and coloring transform (WCT)
- 다양한 스타일화 된 결과를 생성할 수 있음
- 다양한 Style transfer는 주로 **perturbed whitening and coloring transform(PWCT)** 에 의해 달성됨
- PWCT는 2단계로 구성됨 : **1. whitening transform** **2. perturbed coloring transform**

**1. whitening transform**
- 한 쌍의 content image I_c 와 style image I_s가 주어지면, 먼저 특정 레이어(e.g. Relu_3_1)에서 vectorized VGG feature maps를 추출한다. : **F_c, F_s**
- mean vector m_c를 빼서 F_c를 center(중앙)에 둔다.
- 그런 다음 whitening transform을 사용해 F_c를 F^_c로 변환한다. 이때 feature maps은 서로 관련이 없다.(uncorrelated) 여기서 D_c 및 E_c는 Gram Matrix의 특이값 분해(SVD)에 의해 얻어진다.
   <img src="/img/Diversified Arbitrary Style Transfer_3.PNG"></img>

**2. Perturbed Coloring Transform**
- 먼저 mean vector m_s를 빼서 F_s를 center(중앙)에 둔다.
- [19]에서 사용된 transform은 본질적으로 whitening step의 역이다. 즉 방정식 (6)과 같이 F^c를 transform해서 F_s의 Gram matrix와 동일한 Gram matrix를 만족하는 F^_cs를 얻을 수 있다.
    <img src="/img/Diversified Arbitrary Style Transfer_6.PNG"></img>
- coloring transform의 목표는 F^_cs의 Gram matrix를 F_s와 동일하게 만드는 것이다. 섹션3의 분석에 의하면 이 두 feature maps는 동일한 style-specific feature space를 공유한다. 
- 이론적으로 F^_cs는 많은 수의 possibilities를 가져야하지만 방정식 (6)은 하나만 생산함
- 이러한 솔루션을 가능한 많이 탐색하기 위해 deep feature perturbation을 사용할 것을 제안함
- **Key idea : 방정식 (6)에 orthogonal noise matrix를 통합해서 Gram matrix는 유지하면서 feature F^_cs를 perturb(변경, 섭동)하는 것**
<img src="/img/Diversified Arbitrary Style Transfer_7.PNG"></img>
- 실험을 통해 perturbed coloring transform만 사용하면 품질이 저하될 수 있음을 발견 → content information도 noise matrix에 의해 영향을 받아서
- F^_cs는 주로 style feature로 사용되며 style과 content의 균형을 맞추기 위해 content feature F_c를 섞는다. <img src="/img/Diversified Arbitrary Style Transfer_9.PNG"></img>
- 원래의 품질을 유지하면서 다양성을 높이기 위해 다양성 하이퍼파라미터 λ를 도입해 사용자가 제어함. <img src="/img/Diversified Arbitrary Style Transfer_8.PNG"></img>

**Multi-level Stylization**
[19]에 사용된 multi-level  coarse-to-fine stylization을 따르지만 WCT를 PWCT로 대체함. 사실 모든 레벨에 노이즈를 추가할 필요는 없음(5.2절에서 논의)

## 5. Experimental Results
- 기존의 세 가지 WCT-based methods에 본 논문에서 제안하는 deep feature perturbation을 통합해서 실험을 진행.
- 그림 2와 3에서 볼 수 있듯이 너무 많은 노이즈를 도입하는 것은 불필요하며 품질을 저하시킴. 또한 런타임도 증가함
- Trade-off between Diversity and Quality : 하이퍼파라미터 λ값이 증가하면 다양성의 정도가 높아지지만 품질이 떨어짐.
- Relation between Diversity and Stylization Strength : 다양성은 스타일화 강도와도 관련됨. 더 작은 스타일화 강도(α values)에 대해 더 큰 다양성 강도(λ)를 설정할 수 있음
