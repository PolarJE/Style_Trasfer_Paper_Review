# Two-Stage Peer-Regularized Feature Recombination for Arbitrary Image Style Transfer (CVPR 2020)

## 0. Abstract
원하는 스타일(desired style)을 묘사하고 있는 set of examples에 대해 스타일화 된 image conditioning을 생성하기 위한 NST(neural style transfer) 모델을 소개한다.
- zero-shot setting 에서도 고화질의 이미지를 생성하고,
- content geometry(기하학)를 더 자유롭게 변경할 수 있다.
- 이는 "a custom graph convolutional layer"를 통해 잠재 공간(latent space)에서 style과 content를 재결합하는 "Two-Stage Peer-Regularization Layer"를 도입함으로써 가능하다.
- (대부분의 기존 방법과는 달리) Perceptual losses를 계산하기 위해 Pre-trained network에 의존하지 않음. 
- RGB 이미지가 아닌 잠재 공간에서 직접 작동하는 "a new set of cyclic losses" 덕분에 완전한 end-to-end 학습이 가능함.
- 모델을 더 간단하게 배포할 수 있고, 추상적이고 예술적인 Neural image generation 시나리오의 문을 열어줌

## 1. Introduction
현재의 NST 알고리즘에는 몇가지 **한계**가 있다.
  1. the original formulation of Gatys et al은 수행되는 각각의 transfer에 대한 새로운 최적화 프로세스가 필요하므로 실제 시나리오 에서는 비현실적이다. (학습된 스타일에 한해서만 전이함. 현실성 떨어짐)<br> 또한, 일반적으로 분류 작업에서 가져온 사전 훈련된 네트워크에 크게 의존하는데 이는 최적이 아니며 구조 보다는 텍스쳐에 편향되어 있다. 
        
        → 한계를 극복하기 위해 심층 신경망(deep neural networks)은 단일 피드 포워드 단계(single feed forward step)에서 **긴 최적화 절차를 근사화하여 모델을 실시간 처리에 적합하게** 만들도록 제안되었다.    
      ```
     <관련 레퍼>
     [15] Johnson et al. "Perceptual losses for real-time style transfer and super-resolution."
     [35] Ulyanov et al. "Texture networks: Feed-forward synthesis of textures and stylized images."
      ``` 
  2. 둘째, 신경망을 사용하여[8]의 계산 부담을 극복하는 경우, 여러 스타일을 네트워크 가중치에 인코딩하는 기존 모델의 제한된 용량으로 인해 **원하는 모든 스타일 이미지에 대한 모델 훈련이 필요하다.** 이는 스타일의 개념을 사전에 정의할 수 없고 예제로부터 추론해야 하는 사용 사례에 대한 방법의 적용 가능성을 크게 축소한다. 이 두 번째 제한과 관련하여, 최근 연구는 추가로 입력된 이미지의 스타일을 일반화 할 수 있도록  style과 content를 feature space(latent space)에서 분리하려는 시도를 했다. → AdaIN[11]
  <br>

NST를 개선하기 위한 **손실함수**도 나타남. 
- Perceptual Loss[8,15] : pre-trained VGG-19 classifier를 사용해 이미지의 high-level features를 포착하기 때문에 일반적으로 사용됨.<br> → 그러나 [9, Imagenet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness, ICLR, 2019]에서 이의제기함
- Cycle-GAN에서 cycle consistent loss 제안 : 입력 이미지와 대상 이미지 간의 일대일 대응이 필요하지 않아서 data annotation의 부담을 덜어줌.
- Image style transfer는 image의 스타일이 local 속성(e.g. 객체의 일반적인 모양 등)과 global 속성(e.g. textures 등)으로 표현되기 때문에 어렵다. 과거에 제안된 lower dimensional latent space에서의 정보 인코딩이 매우 유망한 결과를 보여줌. <br> **→ 따라서 우리는 pixel-wise features의 local aggregation(지역 응집?)과 metric learning을 사용해서 서로 다른 스타일로 분리하여 잠재 공간에서 이 계층을 모델링하는 것을 지지한다.** 우리가 아는한 이것은 이전 접근 방식에서 명시적으로 해결되지 않음.
- 스타일과 콘텐츠가 완전히 분리된 잠재 공간이 있는 경우, 디코더 가중치에 변환(transformation)을 저장할 필요 없이 input과 conditioning style images사이의 잠재 공간에서 스타일 정보를 교환하여 전송(transfer)을 쉽게 수행할 수 있다. 이러한 접근 방식은 feature normalization과 독립적이며 problematic pre-trained models이 필요하지 않음.
- **그러나 이미지의 content와 style은 완전하게 분리할 수가 없다.** 이미지의 content는 어떤 스타일로 칠해졌는지에 따라 기하학적 변화를 보인다. 
- 최근에 Kotovenko et al. [20, A content transformation block for image style transfer]은 모델이 두 단계로 훈련되는 adversarial setting 에서 a content transformer block 을 제안함<br> : 먼저 style transformer block이 최적화된 후 고정되고 content transformer block이 최적화되어 주어진 스타일과 관련된 geometry의 변화를 설명하는 방법을 학습한다. 따라서 style exchange 는 2단계이고 이러한 의존성을 모델링하는 것은 시각적 결과를 드라마틱하게 향상시킨다.
- **본 논문은 입력 스타일 이미지에서 전역 및 로컬 스타일 콘텐츠를 재조합할 수 있는 *새로운 기능 정규화 계층(feature regularization layer)을* 도입하여 임의 도메인에서 전송을 허용하고 도전적인 제로샷 스타일 전송 시나리오를 다루기 위해 스타일이 외부적으로 정의된 NST 설정을 다룬다.** <br> → geometric deep learning (GDL)에서 아이디어 빌림&잠재 공간의 feature maps에서 peers의 pixel-wise graph를 모델링하여 달성함 <br>

**Contributions**
- 잠재 공간에서 스타일과 콘텐츠를 재조합하는 custom graph convolutional layer를 이용한 NST를 위한 최첨단 접근법
- 사전 훈련된 모델(예: VGG)이 perceptual loss를 계산할 필요 없이 end-to-end training을 가능하게 하는 기존 loss의 새로운 조합
- content 와 style information을 위한 globally-and-locally-combined latent space 구축 및 metric learning을 통해 structure을 도입

## 2. Background
#### Image-Optimization-Based Online Neural Methods.
Content는 VGG-19의 low-level layer로 표현됨.
Style은 several higher layers의 combination of activations로 표현됨.

#### Image-Optimization-Based Offline Neural Methods.
One-Style-Per-Model methods : 각각의 새로운 target style에 대해 별도의 모델을 학습시켜야 함. 동적이고 빈번한 사용에는 다소 비실용적임.
Batch Normalization (BN)보다 Style Transfer에 더 적합한 Instance Normalization (IN) 도입.
- Multiple-Styles-Per-Model methods : 각각의 스타일마다 작은 수의 파라미터를 할당하려고 함.
  - Conditional Instance Normalization (CIN)

- Arbitrary-Style-Per-Model methods : non-parametric 한 방식으로 style information을 처리
  - Adaptive Instance Normalization (AdaIN)
  - Whitening and Coloring Transformations (WTC) : AdaIN의 변형
  - ZM-Net [38] : zero-shot style transfer
  - Avatar-Net [32] : 스타일 이미지에서 파생된 input style features이 semantically 정렬하여 content features가 re-create 하기 위해 "style decorator"의 사용을 제안함.

- Other methods.
  - Cycle-GAN [43] : input & target pairs가 필요 없다. 그러나 여전히 스타일 당 하나의 모델이 필요함.
  - Combo-GAN [1] : 더 확장시켜서 실용적인 멀티 스타일 전송이 가능. 그러나 이 방법도 decoder-per-style이 필요함.
  - Kotovenko et al. [20] : trained style transfer model을 특정한 content로 finetune하는 추가 서브 네트워크인 "content transformer block"을 설계 
  - MUNIT [12], FUNIT [27]
  - Kotovenko et al. [19] : latent space에서 다양한 스타일의 분리를 위해 "fixpoint triplet loss"를 제안. 단일 모델 내에서 두 개의 서로 다른 스타일을 분리하는 방법을 보여줌.

## 3. Method
- 본 논문의 핵심은 semantic content는 보존하면서 input과 target style images을 교환하는 region-based mechanism이다. (StyleSwap [4]과 유사함).
- 스타일과 콘텐츠 정보를 잘 분리해야 한다. 
- 서로 다른 스타일 간의 분리를 위해 metric learning을 사용한다. 이는 디코더가 유지하고 있는 스타일 종속 정보의 양을 크게 줄인다는 것이 실험적으로 입증됐다. 
- 또한, 특정 스타일에 얽매인 content의 기하학적 변화를 설명하기 위해 style transfer를 2단계 프로세스로 모델링 하고 먼저 스타일 전송을 수행한 다음 콘텐츠의 기하학적인 구조를 수정한다. 이는 **Two-stage Peer-regularized Feature Recombination (TPFR) module**을 사용하여 수행된다.

## 3.1. Architecture and losses
<img src="/img/Arbitrary Image Style Transfer.PNG"></img><br/>
- main decoder가 weights로 stylization을 인코딩하는 것을 방지하기 위해 auxiliary decoder[7]가 사용되고, 인코더와 디코더의 파라미터가 훈련 동안에 독립적으로 최적화하기 위해 사용된다.
- Fig 2의 노란색 모듈은 input을 reconstruct(재구성)하기 위해 autoencoder (AE) [29, 41, 28]로 훈련되었다.
- 대신 녹색 모듈은 고정된 파라미터와 노란색 모듈의 인코더를 사용하여 stylized version을 생성하기 위해 GAN으로 훈련된다.
- 두 모듈의 최적화는 판별자와 함께 interleaved 된다.
- 또한 Martineau et al. [16]에 의하면 상대론적 평균 GAN[the Relativistic Average GAN (RaGAN)]은 우리의 적대적 손실 공식으로 사용되며, 이는 전통적인 GAN loss보다 더 안정적이고 더 자연스러운 이미지를 생성하는 것으로 나타난다.

#### Encoder.
모든 input images의 latent representation을 생성하는데 사용되는 인코더는 다운 샘플링을 위한 strided convolutional layers와 여러 ResNet blocks로 구성된다.
latent code z는 2부분으로 구성된다. 
- (z)_c : content part. 이미지 콘텐츠(objects, position, scale 등) 정보를 갖고 있음
- (z)_s : style part. 콘텐츠가 표현되는 스타일(예:level of detail, shapes, 등)을 인코딩. 두 부분으로 나뉜다. 
  - (z)^loc_S : 각각의 feature map의 픽셀마다의 local style information을 인코딩
  - (z)^glob_S : 작은 sub-network를 통해 추가적인 dowm-sampling을 거쳐 feature map마다 single value를 생성한다.

#### Auxiliary decoder.
latent representaion로부터 이미지를 재구성하며 인코더 모듈을 훈련하기 위해 훈련 동안에만 사용된다. 원본 이미지를 재구성하기 위해 몇 개의 ResNet 블록과 fractionally-strided convolutional layers로 구성된다. loss는 [31, 19]에서 영감을 받아 다음과 같은 부분으로 구성된다. 
**- A content feature cycle loss : 동일한 콘텐츠를 나타내는 latent codes를 하나로 묶는 손실** <br><img src="/img/Arbitrary Image Style Transfer_2.PNG"></img><br/>

**- A metric learning loss : latent representations의 style part 강제 clustering 하는 손실** <br><img src="/img/Arbitrary Image Style Transfer_3.PNG"></img><br/>

**- A classical reconstruction loss : autoencoders에서 사용한다. 모델이 inputs을 완벽하게 재구성하도록 학습시키는 손실** <br>
<img src="/img/Arbitrary Image Style Transfer_4.PNG"></img><br/>

**-  a latent cycle loss : inputs의 latent codes를 재구성된 이미지(reconstructed images)의 latent codes와 동일하게 만드는 손실. 훈련을 안정화시킴.**<br>
<img src="/img/Arbitrary Image Style Transfer_5.PNG"></img><br/>

**- total auxiliary decoder loss**
<br><img src="/img/Arbitrary Image Style Transfer_6.PNG"></img><br/>

#### Main decoder.
auxiliary decoder의 architecture를 복제하고 Two-stage  Peer-regularized Feature Recombination module (see Section 3.2)의 output을 사용한다.
main decoder를 훈련하는 동안 encoder는 고정되어 있으며 decoder는 다음 부분으로 구성된 손실 함수를 사용하여 최적화된다.

**1. decoder adversarial loss : P는 real data 분포, Q는 generated(fake) data 분포, C는 discriminator.**
   <br><img src="/img/Arbitrary Image Style Transfer_7.PNG"></img><br/>

**2. transfer latent cycle loss : stylization을 시행하기 위해 latent codes의 content 부분을 보존하면서 latent codes의 스타일 부분을 재결합하여 대상(target) 스타일 클래스를 나타내기 위해 사용한다.**
<br><img src="/img/Arbitrary Image Style Transfer_8.PNG"></img><br/>

**3. the classical reconstruction loss : main decoder가 original inputs을 재구성하는 방법을 배우게 하기 위해 보조 디코더에서도 했던 것처럼 고전적인 재구성 손실을 사용한다.**
<br><img src="/img/Arbitrary Image Style Transfer_9.PNG"></img><br/>

위의 내용을 종합해서 main decoder loss L_D를 구성하면 다음과 같다. 
<br><img src="/img/Arbitrary Image Style Transfer_10.PNG"></img><br/>

#### Discriminator 
channel 차원에 연결된 두 개의 이미지를 받고 N X N의 예측 맵을 생성하는 컨볼루션 네트워크이다. 첫 번째 이미지는 구별(discrimi하는 반면, 두 번째 이미지는 style class의 conditioning 역할을 한다. 두 입력이 동일한 스타일 클래스에서 나오는 경우 출력 예측(prediction)은 이상적인 1이고, 그렇지 않은 경우 0이다.
**Discriminator loss :**
<br><img src="/img/Arbitrary Image Style Transfer_11.PNG"></img><br/>

## 3.2. Two-stage Peer-regularized Feature Recombination (TPFR)
TPFR 모듈은 PeerNets [34]와 Graph Attention Layer (GAT) [37]에서 영감을 얻어 콘텐츠와 스타일 정보의 분리를 활용하여 latent space에서 스타일 전송을 수행한다. (수식 2 및 3에 의해 적용됨). 피어 정규화 된(Peer-regularized) feature 재조합은 다음 paragrap에 설명 된대로 두 단계로 수행된다.
- Style recombination. : 
  - z_i와 z_t를 입력으로 받고 동료 그래프(graph of peers)를 유도하기 위해 유클리드 거리를 사용해서 (z_i)_C와 (z_t)_C 사이의 k-NN을 계산한다.<br> 
  - 그래프 노드에 대한 **Attention 계수(coefficients)** 는 (z_out)_S의 스타일 부분을 가까운 이웃 표현의 convex combination(볼록 조합)으로 재조합하기 위해 계산되고 사용된다. <br> 
  - latent code의 콘텐츠 부분은 바뀌는 대신 유지된다.
  - z_out = [(z_i)_C, (z_out)_S].
  - feature map m의 픽셀 (z_m)_C가 주어지면, 모든 peer feature maps n_k의 모든 픽셀의 d차원 feature maps 공간에서 k-NN 그래프가 고려된다.
  - 픽셀에 대한 스타일 부분 (z)_S의 새로운 값은 다음과 같이 표현된다. 
 <br><img src="/img/Arbitrary Image Style Transfer_12.PNG"></img><br/>
 <img src="/img/Arbitrary Image Style Transfer_13.PNG"></img><br/>

- Content recombination. :
  - style latent code가 재결합되면 유사한 프로세스가 반복되어 새로운 스타일 정보에 따라 content latent code를 변환한다. 
  - 이 경우 입력 z_out = [(z_i)_C, (z_out)_S] 및 z_t = [(z_t)_C, (z_t)_S]로 시작하고 k-NN 그래프는 주어진 style latent codes (z_out)_S, (z_t)_S로 계산된다.
  - 이 그래프는 attention coefficients를 계산하고 content latent code를 (z_final)_C로 재결합하기 위해 방정식 12와 함께 사용된다. 
  - TPFR 모듈의 output은 그러므로 latent code의 스타일과 콘텐츠 부분을 재결합하는 새로운 latent code z_final = [(z_final)_C, (z_out)_S]이다.

## 4. Experimental setup and Results (정성평가만 함)
## 4.1 Training
- dataset[31] : https://github.com/CompVis/adaptive-style-transfer / 13개의 target style (4,430 paintings) / 관련있는 클래스의 Places365 dataset으로부터 얻은 real photo images(624,077)
- optimization scheme : ADAM[18]
- 200 epochs (lr : 4e-4, batch size : 2)
- After 50 epoch (lr : linearly to zero, batch size : 임의로 지정가능, 논문에선 2로함)
- training 동안 256X256 해상도로 크기가 조정됨.
- test 할 때, 임의의 크기의 이미지에서 가능하다.
  
## 4.2 Style Transfer
기존의 다른 모델들과는 달리 본 논문의 네트워크는 각각의 스타일에 대한 재훈련이 필요하지 않고, 이전에 본 적 없던 스타일에서도 transfer를 할 수 있다.
<img src="/img/Arbitrary Image Style Transfer_fig5.PNG"></img><br/>
- Zero-shot style transfer.
- 훈련 중에 사용되는 auxiliary decoder는 solutions(해)를 퇴화시키는 것을 방지하기 때문에 중요하다.
- main decoder로 end-to-end로 직접 encoder를 훈련시키지 않음.
- latent code를 content와 style로 분리하면 소개한 2단계 style transfer가 가능하며 피카소와 같은 스타일을 위해 객체(objects)의 형태(shape) 변화를 고려하는 것이 중요하다.
- Two-stage recombination은 다양한 스타일에 대해 더 나은 일반화를 제공한다. content features에 기반한 style 교환만 수행하는 것은 일부 경우에 완전히 실패한다.
- metric learning 은 style latent space 에서 더 나은 clustering을 적용하고, stylized image에서 몇 가지 중요한 details를 향상시킨다.
- 결합된 local and global style latent code는 edges 및 brushstrokes의 변화를 적절히 설명하는데 중요하다.

## 5. Conclusions
- 본 논문은 다양한 한계를 완화하고 도전적인 zero-shot transfer setting에서도 사용할 수 있는 neural style transfer을 위한 새로운 모델을 제안한다.
- 이는 graph convolutions을 사용하여 latent representation의 스타일 구성요소를 재조합(recombine)하는 **Two-Stage Peer-Regularization Layer**와
- feature space의 cycle consistency과 결합된 다른 스타일의 분리를 강제하는 **metric learning loss** 덕분이다.
- degenerate solutions (퇴화된 솔루션?)를 방지하고 생성된 샘플의 충분한 가변성을 적용하기 위해 **auxiliary decoder**가 도입된다.
- 그 결과 perceptual loss를 계산하기 위해 사전훈련된 모델 없이도 end-to-end 훈련을 할 수 있는 최첨단 방법이며, 따라서 NST에 대한 그러한 features의 신뢰성에 대한 최근의 우려를 완화시킨다.
- 더 중요한 것은 각 input 및 target 쌍에 대해 디코더 및 인코더가 필요한 많은 경쟁 방법과 달리 **arbitrary styles 간에 전송을 수행하기 위해 단일 인코더와 단일 디코더만 필요로 한다**는 것이다.
- 이는 사용자가 자신만의 스타일을 정의하는 real-world image generation scenarios에 적용할 수 있게해준다.
