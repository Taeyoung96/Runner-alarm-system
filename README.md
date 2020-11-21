# 딥러닝 기반 러너 알람 시스템

광운대학교 로봇학부 학술소모임 **'BARAM'** 20년도 전반기 Toy Project에 대한 소스코드입니다.  

## 개발 환경
|OS|사용 언어|사용 IDE|
|:---:|:---:|:---:|
|Ubuntu 18.04|Python|Pycharm|

## 프로젝트 개발 동기

- 최근 딥러닝에 대한 관심이 증가하면서 Object Detection Model을 이용하여 Toy project를 진행하고 싶었다.  
  Object Detection Model 중 'Yolo-v3'가 성능도 좋고 빠르게 동작한다는 것을 알게 되었고,  
  이를 이용하여 아이들의 뛰는 모습과 걷는 모습을 판별하는 모델을 만들어 보고 싶었다.  
  CCTV와 같은 영상을 이용하여 아이들이 뛰거나 걷는 것을 판단하고,  
  만약 뛰는 애들을 인식할 경우 경고음을 내어 경각심을 주도록 하는 것이 목표이다.  

## 프로젝트 개요
1. [Yolo_mark](https://github.com/AlexeyAB/Yolo_mark)를 통한 Data annotation 작업(약 3000장).  
2. [Darknet](https://github.com/AlexeyAB/darknet)을 이용한 Training.  
3. 'Pydub' 라이브러리를 이용하여 경고음 추가.  
4. 멀티 쓰레드를 이용하여 영상과 경고음이 충돌하는 것을 막음.  

### Code Overview  
- `yolov3.py` : main 소스 파일의 역할. OpenCV dnn 모듈을 이용하여 Yolo 모델을 읽어온다.  
- `yolov3-make.cfg` : 나만의 yolo 모델에 대한 cfg 파일.  
- `weights/` : Trasfer learning을 이용하여 만든 weight 파일 모음.  
- `manual/` : Yolo_mark를 이용할 때 참고한 자료 모음.  
- `sounds/` : 사용한 경고음 모음 폴더.  
- `video/` : Test용 video 모음 폴더.  

### Project scenario

1. 아나콘다를 이용하여 개발환경과 동일하게 환경을 설정해준다.  
  - `conda create -n runnerAlarmSystem python==3.7`
  - `conda activate runnerAlarmSystem`  
  - `pip install opencv-python`  
  - `pip install numpy`  
  - `pip install pydub`  
2. `git clone`을 이용하여 소스 코드를 다운받고 터미널로 그 경로로 들어간다.  
  - `git clone https://github.com/Taeyoung96/Runner-alarm-system.git`
  - `cd Runner-alarm-system`  
3. `yolov3.py`를 자신의 경로와 맞게 수정한다. (`gedit yolov3.py`)  
  - `weights/`에서 자신이 사용할 weight 파일을 `yolov3.py`와 같은 경로로 설정.  
  - `warning.wav`을 자신의 폴더의 절대경로로 설정.  
  - 웹켐 번호 및 동영상을 자신의 폴더에 맞게 설정.  
4. 터미널에서 `python yolov3.py`를 실행한다.

## 프로젝트 결과

<p align="center"><img src="https://user-images.githubusercontent.com/41863759/99877923-e36e7f80-2c44-11eb-834d-705c2e6998b0.gif" width="500px"></p>  
<p align="center"> 'test1.mp4'로 Test한 결과 영상 </p>  


