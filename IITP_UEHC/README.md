# Acoustic_echo_cancellation
Carnival system을 구성하는 Acoustic echo cancellation 모델입니다. 과학기술통신부 재원으로 정보통신기획평가원(IITP) 지원을 받아 수행한 "원격 다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제 공개 코드입니다. (2021.05~2024.12)

본 모델은 2차년도 시스템을 통합하여 에코 및 하울링 통합 제거 모델로 여러 개의 마이크에 입력되는 음성과 far-end 신호를 이용하여 에코 및 하울링을 제거하는 형태로 동작합니다. 본 실험은 SiTEC 한국어 음성 DB를 사용하여 진행되었습니다.

본 모델의 특징은 다음과 같습니다.
* Far-end단의 system을 고려하여 closed-loop의 실제 화상회의 환경을 시뮬레이션
* 위로 인하여 far-end의 신호는 far-end clean speech가 아닌 제거하지 못한 near-end의 신호도 포함
* 에코 및 하울링 통합 제거 모델이 없을 경우 하울링 발생
* 멀티 마이크 기반으로 spatial que를 이용
* 학습 시, 매 프레임마다의 학습은 비효율적이므로 지도학습을 통한 학습 데이터 미리 생성
* 테스트 시, 매 프레임마다의 동작으로 causality에 따른 불러오는 모델 파일명 및 STFT 파일명이 다름

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

## Training echo cancelltaion model
To train the model, run this command

    python train.py
    
## Making UEHC train dataset
To generate UEHC DB, run this command

    python make_finetune_data.py

## Training UEHC model
To train the model, run this command

    python train_UEHC.py
    
## Test
To test the model, run this command

    python test_UEHC.py
    
## Reference code
* AEC-Challenge : https://github.com/microsoft/AEC-Challenge
