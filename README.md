# Acoustic_echo_and_feedback_cancellation
AEC-Challenge baseline을 개선하여 context를 고려한 GRU 기반의 에코제거 모델입니다. 본 코드는 2021년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "원격 다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제의 일환으로 공개된 에코제거의 1차년도 코드입니다.

본 모델의 특징은 다음과 같습니다.
* Audio context를 고려한 GRU 기반의 네트워크
* 네트워크의 feature로 STFT를 사용

Microsoft에서 개최한 AEC-Challenge의 synthetic DB로 훈련하였으며 총 10,000 set 중 500 set는 validation 및 test set으로 사용하고 나머지 9,500 set는 train set로 사용하였습니다.

## Requirements
* Pytorch
* numpy
* soundfile
* librosa
* scipy

## Prepare dataset
You can use [AEC-Challenge synthetic dataset](https://github.com/microsoft/AEC-Challenge/tree/main/datasets/synthetic)

## Training
To train the model, run this command

    python train.py -c configs/train.json
    
## Test
To test the model, run this command

    python test.py -c configs/test.json
    
## References
*Microsoft AEC-Challenge:(https://github.com/microsoft/AEC-Challenge)
