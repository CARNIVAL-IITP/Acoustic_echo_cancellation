# Acoustic_echo_and_feedback_cancellation
1차년도 모델을 개선하여 global context를 고려한 LSTM 기반의 에코제거 모델입니다. 본 코드는 2022년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "원격 다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제의 일환으로 공개된 에코제거의 2차년도 코드입니다.

본 모델의 특징은 다음과 같습니다.
* Audio context를 고려한 LSTM 기반의 네트워크
* 네트워크의 feature로 STFT를 사용
* 신호간의 correlation을 loss로 사용하여 개선된 성능 확보

본 과제의 DB 수집에 따라 생성된 DB로 훈련하였으며 SiTEC dataset을 이용하였습니다.
Train/ validation set을 SiTEC Dict 02 dataset을 이용하였으며 test set은 SiTEC Dict 01 dataset을 이용하였습니다.
각각의 파일은 8초 길이이며 각 8개의 방에 대하여 방별로 train 4,500 set (10 h), validation 500 set (약 1.11 h), test 500 set (약 1.11 h) 분량입니다.

## Requirements
* Pytorch
* numpy
* soundfile
* librosa
* scipy

## Prepare dataset
We used [SiTEC dataset](http://sitec.or.kr)
Train 데이터의 생성은 SiTEC Dict 02 dataset 중에서 각각 임의의 남녀 화자 100명씩을 고르고 매칭을 통해 near-end/far-end 쌍으로 남/여 20, 여/남 20, 남/남 30, 여/여 30쌍을 구축하고 far-end 화자의 발화 중 임의의 3개를 골라 이어붙여 far-end를 생성하고 near-end 화자의 발화 중 임의의 1개를 골라 양쪽으로 zero padding을 통해 길이를 맞추어 진행하였습니다.
Validation 데이터의 생성은 SiTEC Dict 02 dataset 중에서 학습에 사용하지않은 임의의 남녀 화자 20명씩을 고르고 매칭을 통해 near-end/far-end 쌍으로 남/여 5, 여/남 5, 남/남 5, 여/여 5쌍을 구축하였으며 그 외는 학습 데이터의 생성과 동일하게 진행하였습니다.
Test 데이터의 생성은 SiTEC Dict 01 dataset 중에서 각각 임의의 남녀 화자 30명씩을 고르고 매칭을 통해 near-end/far-end 쌍으로 남/여 8, 여/남 8, 남/남 7, 여/여 7쌍을 구축였으며 그 외는 학습 데이터의 생성과 동일하게 진행하였습니다.

<!-- ## Training
To train the model, run this command

    python train.py -c configs/train.json
    
## Test
To test the model, run this command

    python test.py -c configs/test.json -->
