# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
#Load and play wav file
from pydub import AudioSegment
from pydub.playback import play

import threading

#WORKERS = 300
audio = AudioSegment.from_wav('warning.wav') #2020.06.12 소리파일을 바꾸고 싶으면, ()에 있는 파일명을 바꿔주시면 됩니다.
######################################################
#원하는 소리 파일을 'warning1.wav'이거 대신에 넣어주세요.#
######################################################

class Worker(threading.Thread):
    def __init__(self):
        self.flag = 0
        super(Worker, self).__init__()
        

    def run(self):
        play(audio)

# Load Yolo
#net = cv2.dnn.readNet("darknet19_448.conv.23", "yolov3_obj.cfg")
net = cv2.dnn.readNet("yolov3-make_last_v2.weights", "yolov3-make.cfg")
######################################################################
#원하는 weights 파일을 "yolov3-make_last.weights" 대신에 넣어주세요.   #
######################################################################
classes = []
with open("obj.names", "r") as f:
	classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# Loading camera
cap = cv2.VideoCapture('/home/min/Desktop/Yolov3/video/test1.mp4')    # 2020.06.12 웹캠 번호 바꾸기 웹캠 번호가 0일수도, 1일수도 그 이상일 수도 있다. '/home/taeyoung/darknet/data/student.mp4'
#############################################################################
#웹켐 번호 및 동영상 재생을 하려면 cv2.VideoCapture(2) 안에 2를 바꾸면 됩니다.   #
#############################################################################

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
wList = []
for i in range(10):
    w = Worker()
    wList.append(w)
ti=0
L = []
while True:
    _, frame = cap.read()
    frame_id += 1
    height, width, channels = frame.shape
 # Detecting objects
    # 320*320 #416*416 #609*609 <=== 정확도 조절
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)   # width of object
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)       #the starting X position of detected object
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))   #percentage
                class_ids.append(class_id)#the name of detected object
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)
    
            if class_ids[i] == 1:
                L.append(class_ids[i])
   
    if(frame_id % 10==0 and  L.count(1)>8): #2020.06.12 Run_counting을 현재 5로 설정되어 있는데, 상황에 따라 바꿔야 함.
################################################################
#frame_id는 몇 프레임 마다 한번씩 체크를 할껀지 정하는 변수 입니다. 지금은 30프레임마다 한번씩 검사를 하고 있어요.
#L.count(1)은 30프레임 마다 검사를 했을 때, Run이라고 판단하는 변수가 몇 이상일 때 경보음을 울릴 것인가 경계값을 정하는 변수 입니다.
###################################################################
        w = Worker()
        w.start()
        ti+=1
        L.clear()


    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

wList[ti].join()
cap.release()
cv2.destroyAllWindows()
