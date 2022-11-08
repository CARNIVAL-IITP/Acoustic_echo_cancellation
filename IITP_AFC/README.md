# Acoustic feedback cancellation  
본 코드는 2022년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "원격 다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제의 일환으로 공개된 하울링 제거의 2차년도 코드입니다.  
  
본 모델의 특징은 다음과 같습니다.
* Audio context와 local 정보를 고려하기 위한 CRN 기반의 네트워크
* 네트워크의 feature로 STFT를 사용
* 신호간의 correlation을 loss로 사용하여 개선된 성능 확보
  
본 과제의 DB 수집에 따라 생성된 DB로 훈련하였으며 SiTEC dataset을 이용하였습니다.  
SiTEC Dict 01 dataset의 남녀 화자 각 200명 중 남녀 각 170명 (총 340명)으로 train dataset을, 남녀 각 15명으로 validation dataset을, 나머지 남녀 각 15명으로 test set을 구성하였습니다.  
또한 크기 및 reverberation time이 다른 총 8개의 직육면체의 room environment를 설정하여 시스템을 구성하였습니다.  
  
## Requirements
* Pytorch
* numpy
* soundfile
* librosa
  
## Prepare dataset
We used [SiTEC dataset](http://sitec.or.kr)

Train 데이터의 생성은 SiTEC Dict 01 dataset 중 남녀 화자 각 170명씩을 고르고 8개의 방환경에 대해서 하울링 시스템을 통해 하울링 발생 신호 및 하울링 발생 전 신호를 생성하였습니다.  
Validation 데이터의 생성은 SiTEC Dict 01 dataset 중 남녀 화자 각 15명씩을 고르고 8개의 방환경에 대해서 하울링 시스템을 통해 하울링 발생 신호 및 하울링 발생 전 신호를 생성하였습니다.  
Test 데이터의 생성은 SiTEC Dict 01 dataset 중 나머지 남녀 화자 각 15명씩을 고르고 8개의 방환경에 대해서 하울링 시스템을 통해 하울링 발생 신호 및 하울링 발생 전 신호를 생성하였습니다.  

