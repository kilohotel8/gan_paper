# UNIT

## Backgrounds

- image to image translation problem은 supervised learning과 unsupervised learning으로 풀 수 있다.
- 쌍으로 대응하는 이미지를 찾기 어렵기 때문에 unsupervised learning이 범용성이 높다.
- 하지만 unsupervised learning 상황에서 서로 다른 도메인에 있는 marginal distribution에서 joint distribution을 추론하기가 어렵다.(수많은 가능성이 있기 때문에)
- 이를 해결하기 위해서 shared-latent space assumption을 사용한다.

## Shared-latent space assumption

- 임의의 x<sub>1</sub>과 x<sub>2</sub>라는 이미지가 있을때 shared-latent space에 두 이미지를 복원할 수 있는 shared latent code인 z가 있다고 가정한다.
- Shared-latent space를 사용하기 위해서 shared intermediate representation h를 추가적으로 가정한다.
