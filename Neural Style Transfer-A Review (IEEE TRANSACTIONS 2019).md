# Neural Style Transfer: A Review (IEEE TRANSACTIONS 2019)

## 0. Abstract
CNN을 사용해 다양한 스타일의 content image를 만드는 프로세스를 Neural Style Transfer (NST) 라고 한다. 본 논문에서는 NST를 향한 현재 진행 상황에 대한 포괄적인 개요를 제공한다.
- 첫째, 우리는 NST 분야의 최근 발전을 조사, 분류 및 요약한다.
- 둘째, 여러 평가 방법을 제시하고 서로 다른 NST 알고리즘을 실험적으로 비교한다.
- 셋째, 이 분야의 현재 과제를 요약하고 향후 작업에서 어떻게 대처할 것인지에 대한 방향을 제안한다.


## 1. Introduction
- 이미지를 자동으로 합성된 예술 작품으로 바꾸는 방법을 탐구하는 많은 연구가 있다.
- 이러한 연구 중 비사실적 렌더링(non-photorealistic rendering, NPR)의 발전은 고무적이며 computer graphics 분야에서 확고하게 자리 잡은 분야다.
-  그러나 대부분의 NPR 스타일화 알고리즘은 특정한 예술적 스타일을 위한 것이고 다른 스타일로 확장하기가 쉽지 않다.
-  Style Transfer란 보통 텍스처 합성의 일반화된 문제로 연구되는데 이는 source로부터 target으로 텍스처를 추출하여 전달하는 것이다.
-  Hertzmann et al.[9]는 제공되는 unstylised 그리고 stylised images의 example pairs로부터 유사한 변환을 학습하여 일반화된(generalised) style transfer를 수행하기 위한  "image analogies"라는 이름의 프레임워크를 제안했다. 그러나 이 방법은 오직 low-level image features만 사용하고 이미지의 구조를 포착하는데에 종종 실패한다는 한계가 존재한다.
- 최근에는 Gatys et al. [10]이 CNN을 사용해서 natural images에 유명한 작품의 스타일을 입히는 방법을 연구한다. 그들은 pre-trained CNN의 feature responses로 사진의 content를 모델링하고 summary feature statistics로 예술 작품의 style을 모델링 할 것을 제안한다. 이는 스타일 유형에 대한 명시적인 제한도 없고, GT 도 필요하지 않기 때문에 이전 연구들의 제약을 깼다. CNN을 사용하는 NST (Neural Style Transfer)라는 새로운 분야를 열었다.
-  Section 2 : CNN을 사용하지 않는 이전의 예술적 렌더링 방법에 대한 이야기
-  Section 3 : NST의 어원(derivation)과 기초(foundation)에 대한 이야기
-  Section 4 : 기존 NST 알고리즘을 분류하고 설명함
-  Section 5 : Section 4의 확장 및 개선 전략
-  Section 6 : NST 알고리즘을 평가하기 위한 방법론 제시
-  Section 7 : NST의 상업적 응용
-  Section 8 : NST 분야의 현재 과제(challenges)
-  Section 9 : 향후 연구를 위한 몇 가지 방향을 설명

## 2. STYLE TRANSFER WITHOUT NEURAL NETWORKS
- NPR (Non-photorealistic Rendering)
- AR (Artistic Rendering)
- IB-AR (image-based artistic rendering)
- SBR (Stroke-Based Rendering) : 하나의 특정한 스타일(ex.유화,수채화,스케치)에 대해서만 설계됨. 임의의 스타일은 불가능
- Region-Based Techniques : 영영 분할(region segmentation)을 통합하여 영역의 content를 기반으로 렌더링을 조정할 수 있도록 하는 것. 이것 역시 임의의 스타일은 불가능
- Example-Based Rendering : 예제 쌍(exemplar pair) 간의 mapping을 배우는 것. 일반적으로 image analogies는 다양한 예술적 스타일에 효과적이다. 그러나 훈련 데이터가 쌍으로 존재해야 해서 실제로 사용할 수는 없고, 또 다른 제한은 low-level features만 사용한다는 것이다. 따라서 일반적으로 content와 style을 효과적으로 캡처하지 못하며 성능이 제한된다.
- **Image Processing and Filtering : 예술적 이미지를 만드는 것은 이미지 단순화와 추상화를 목표로하는 과정입니다. 따라서 주어진 사진을 렌더링하기 위해 관련 이미지 처리 필터를 채택하고 결합하는 것을 고려하는 것은 당연합니다. <br>예를 들어, [18]에서 Winnemoller et al 은 처음으로 bilateral [19]과 Gaussian 필터의 차이 [20]를 이용하여 만화와 같은 효과를 자동으로 생성합니다. 다른 범주의 IB-AR 기술과 비교할 때 이미지 필터링 기반 렌더링 알고리즘은 일반적으로 구현하기 쉽고 실제로 효율적입니다. 그러나 스타일 다양성이 매우 제한적입니다.**
- 요약 : 일반적으로 유연성(flexibility), 스타일 다양성(style diversity), 효과적인 이미지 구조 추출(effective image structure extraction)의 한계가 있었다. 따라서 새로운 알고리즘이 요구되었고 NST가 탄생하게 되었다.

## 3. DERIVATIONS OF NEURAL STYLE TRANSFER (NST의 어원)
스타일은 텍스처와 매우 큰 관련이 있으므로 style transfer의 간단한 방법은 Visual Style Modelling을 이전에 잘 연구된 Visual Texture Modelling 방법과 연결하는 것이다. 스타일 표현을 얻은 후에는 content는 보존하면서 원하는 스타일로 이미지를 재구성 해야하는데 이는 Reconstruction 기술로 해결할 수 있다.

## 3.1 Visual Texture Modelling
1) Parametric Texture Modelling with Summary Statistics <br>
     - **[27] “Texture synthesis using convolutional neural networks, 2015"** 는 CNN 도메인에서 summary statistics를 처음으로 측정했다. 그들은 pre-trained VGG network의 다른 layers에 있는 filter responses 간의 상관관계(correlations)인 model textures에 대한 Gram 기반의 표현을 설계한다. 그러나 Gram 기반의 표현은 global 통계를 캡처하고 공간 배열을 섞도록(?) 설계 되었기 때문에 long-range 대칭 구조로 일반적인 텍스처를 모델링하는데 만족스럽지 못한 결과가 나온다. 
     - **[29] "Incorporating long-range consistency in cnn-based texture generation, 2017"** 는 이 문제를 해결하기 위해 feature maps을 수평 및 수직으로 옮기는 방법을 제안했다. 이 방식은 표현이 공간 배열 정보를 포함하고 있으므로 대칭 속성을 가진 모델링에 더 효과적이다.

2) Non-Parametric Texture Modelling with MRFs <br>
   
## 3.2 Image Reconstruction
추출된 이미지 표현에서 전체 입력 이미지를 재구성하는 역(reverse) 프로세스이다.<br>
1) **Image-Optimisation-Based Online Image Reconstruction** : image space에서 gradient descent를 기반으로 함. 따라서 재구성하는 이미지가 클 때 시간이 많이 걸린다. 이미지를 반복적으로 최적화하여 스타일을 전달함. 
2) **Model-Optimisation-Based Offline Image Reconstruction** : [30]의 효율성 문제를 해결하기 위해 Dosovitskiy와 Brox [31]는 피드 포워드 네트워크를 미리 훈련시키고 훈련 단계에서 계산 부담을 줄 것을 제안합니다. 테스트 단계에서 역방향 프로세스는 네트워크 포워드 패스로 간단히 수행 할 수 있습니다. 그들의 알고리즘은 이미지 재구성 프로세스의 속도를 크게 높입니다. 후반 작업에서 [32], 결과를 개선하기 위해 Generative Adversarial Network (GAN) [33]를 추가로 결합합니다. 즉, 생성 모델을 오프라인으로 최적화하고 단일 정방향 패스로 스타일화된 이미지를 생성한다.

## 4. A TAXONOMY OF NEURAL STYLE TRANSFER ALGORITHMS (NST의 분류체계)

## 4.1 Image-Optimisation-Based Online Neural Methods
먼저 해당 스타일 및 콘텐츠 이미지에서 스타일 및 콘텐츠 정보를 모델링하고 추출한 다음 대상 표현(target representation)으로 재결합(recombine)한 다음 대상 표현(target representation)과 일치하는 스타일화(stylised) 된 결과를 반복적으로 재구성(reconstruct)하는 것이다.

## 4.1.1 Parametric Neural Methods with Summary Statistics
NST의 첫번째 하위 집합은 요약 통계(Summary Statistics)를 사용하는 Parametric Texture Modelling을 기반으로 한다. 스타일은 공간 요약 통계(spatial summary statistics)의 set으로 특정지어진다.
