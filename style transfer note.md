
# Image Style Transfer Using Convolutional Neural Networks (CVPR 2016)
[[Code Practice]](https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Style_Transfer_Tutorial.ipynb) 

### Style Reconstruction과 Content Reconstruction 살펴보기
- style loss는 여러개의 layer를 같이 사용해서 구한다. 깊은 layer까지 함께 사용했을 때, 화풍과 같은 스타일 정보를 더 잘 받아온다. 
- 분류 작업을 위한 CNN의 경우 앞쪽에 있는 layer일수록 채널의 개수는 작고 너비와 높이는 크다. 따라서 앞쪽 layer에서 구한 Gram Matrix는 상당히 크기가 작고, 뒷쪽으로 갈 수록 각 layer의 output으로 나온 feature map의 channel 값이 커짐에 따라 Gram Matrix 크기 또한 커진다.
- Content loss는 특정한 레이어 하나만 설정해서 그 레이어의 아웃풋으로 나온 feature 값 자체가 같아질 수 있도록 업데이트 함. 레이어가 깊어지면 원본 이미지 콘텐츠에 대한 정보를 완전하게 복구하지는 않는 형태로 나옴. 즉 깊은 레이어를 사용할수록 디테일한 픽셀에 대한 정보는 소실 된다. 
<br><br>
- 이전까지의 논문들에는 공통적인 한계점이 존재했다. low-level image features만 활용이 가능했다는 점이다. 여기서 low-level feature 란 이미지의 edge나 corner와 같은 feature를 의미하고, high-level feature 란 이미지에 존재하는 어떠한 object의 일부분이나 혹은 전체 이미지를 포괄하는 feature를 의미할 때 사용하는 말이다. 따라서 이전까지 존재했던 방법들을 사용해서 texture transfer를 진행하면 결과 이미지에서 high-level 특징은 거의 변화가 없는 경우가 많았다. 결과적으로 본 논문에서는 Deep Convolutional Neural Networks를 활용해서 high-level feature를 적절히 추출하고 이를 조절할 수 있도록 해서 Style Transfer가 가능하도록 한다.
- 잘 학습된 CNN을 활용하면 충분히 스타일과 컨텐츠를 분리해서 각각 추출할 수 있다.
