# Acoustic feedback cancellation  
본 코드는 2022년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "원격 다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제의 일환으로 공개된 하울링 제거의 2차년도 코드입니다.  
  
본 모델의 특징은 다음과 같습니다.
* Audio context를 고려한 LSTM 기반의 네트워크
* 네트워크의 feature로 STFT를 사용
* 신호간의 correlation을 loss로 사용하여 개선된 성능 확보
