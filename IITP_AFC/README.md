# Acoustic feedback cancellation  
Carnival system을 구성하는 Acoustic feedback cancellation 모델입니다. 과학기술통신부 재원으로 정보통신기획평가원(IITP) 지원을 받아 수행한 "원격 다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제 공개 코드입니다. (2021.05~2024.12)  

본 모델은 1차년도 모델을 개선하여 global context를 고려한 LSTM 기반의 하울링 제거 모델로 단일 마이크에 입력되는 음성을 이용하여 하울링을 제거하는 형태로 동작합니다. 본 실험은 SiTEC 한국어 음성 DB를 사용하여 진행되었습니다.  

본 모델의 특징은 다음과 같습니다.
* Audio context와 local 정보를 고려하기 위한 CRN 기반의 네트워크
* 네트워크의 feature로 STFT를 사용
* 신호간의 correlation을 loss로 사용하여 개선된 성능 확보  
  
Train/ validation set을 SiTEC Dict 02 dataset을 이용하였으며 test set은 SiTEC Dict 01 dataset을 이용하였습니다.  
각각의 파일은 8초 길이이며 각 8개의 방에 대하여 방별로 train 4,500 set (10 h), validation 500 set (약 1.11 h), test 500 set (약 1.11 h) 분량입니다.
본 코드는 2022년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "원격 다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제의 일환으로 공개된 하울링 제거의 2차년도 코드입니다.  

SiTEC Dict 01 dataset의 남녀 화자 각 200명 중 남녀 각 170명 (총 340명)으로 train dataset을, 남녀 각 15명으로 validation dataset을, 나머지 남녀 각 15명으로 test set을 구성하였습니다.  
각각의 파일은 약 4초 길이이며 각 8개의 방에 대하여 방별로 train 282,528 set (313 h), validation 24,968 set (약 27 h), test 25,824 set (약 28 h) 분량입니다.
본 코드는 2022년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "원격 다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제의 일환으로 공개된 하울링 제거의 2차년도 코드입니다.  

## Requirements
* Pytorch
* numpy
* soundfile
* librosa
* pystoi
  
## Prepare dataset
We used [SiTEC dataset](http://sitec.or.kr)

Train 데이터의 생성은 SiTEC Dict 01 dataset 중 남녀 화자 각 170명씩을 고르고 8개의 방환경에 대해서 하울링 시스템을 통해 하울링 발생 신호 및 하울링 발생 전 신호를 생성하였습니다.  
Validation 데이터의 생성은 SiTEC Dict 01 dataset 중 남녀 화자 각 15명씩을 고르고 8개의 방환경에 대해서 하울링 시스템을 통해 하울링 발생 신호 및 하울링 발생 전 신호를 생성하였습니다.  
Test 데이터의 생성은 SiTEC Dict 01 dataset 중 나머지 남녀 화자 각 15명씩을 고르고 8개의 방환경에 대해서 하울링 시스템을 통해 하울링 발생 신호 및 하울링 발생 전 신호를 생성하였습니다.  
## Training
To train the model, run this command

    python 1_AFC_train.py
    
## Test
To test the model, run this command

    python 2_AFC_enhancement.py

## Evaluation
To be added

## Reference code
To be added
