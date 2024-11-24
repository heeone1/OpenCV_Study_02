import cv2
import sys
import numpy as np

def construct_yolo_v3():
    f = open('coco_names.txt', 'r')
    class_names=[line.strip() for line in f.readlines()]

    model=cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
    # 학습모델(네트워크) 불러오기 cv2.dnn.readNet(model, config=None, framework=None) -> retval
    # • model: 훈련된 가중치를 저장하고 있는 이진 파일 이름
    # • config: 네트워크 구성을 저장하고 있는 텍스트 파일 이름, config가 없는 경우도 많습니다.
    # • retval: cv2.dnn_Net 클래스 객체
    layer_names=model.getLayerNames()
    print(layer_names)      # <1>
    out_layers=[layer_names[i-1] for i in model.getUnconnectedOutLayers()]
    # 신경망의 출력을 담당하는 3개의 층, yolo_82, yolo_94, yolo_106
    print(out_layers)       # <1>

    return model,out_layers,class_names

#테스트 하는 과정
def yolo_detect(img,yolo_model,out_layers):
    height,width=img.shape[0],img.shape[1]
    test_img=cv2.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True) #크기도 (448,448)로 정규화
    # 네트워크 입력 블롭(blob) 만들기 cv2.dnn.blobFromImage(image, scalefactor=None, size=None, mean=None, swapRB=None) -> retval
    # • image: 입력 영상
    # • scalefactor: 입력 영상 픽셀 값에 곱할 값. 기본값은 1.
    # • size: 출력 영상의 크기. 기본값은 (0, 0).
    # • mean: 입력 영상 각 채널에서 뺄 평균 값. 기본값은 (0, 0, 0, 0).
    # • swapRB: R과 B 채널을 서로 바꿀 것인지를 결정하는 플래그. 기본값은 False.
    # • retval: 영상으로부터 구한 블롭 객체. numpy.ndarray. shape=(N,C,H,W). dtype=numpy.float32.
    yolo_model.setInput(test_img) # 테스트 이미지를 YOLO 학습모델의 새 입력 값으로 설정
    output3=yolo_model.forward(out_layers) # YOLO 학습모델의 out_layers로 출력을 계산하기 위해 정방향으로 전달
    # output3 객체는 14*14*3*85 텐서, 28*28*3*85 텐서, 56*56*3*85 텐서를 리스트

    box,conf,id=[],[],[]		# 박스, 신뢰도, 부류 번호
    for output in output3:
        #print(len(output))
        for vec85 in output:
            # vec85는 (x,y,w,h,o,p1,p2,⋯,p80)을 표현
            # vec85[5:]는 앞 4개는 바운딩 박스 정보, 5번째는 물체일 가능성, 이후 80개는 물체 부류 확률
            scores=vec85[5:]    # p1,p2,⋯,p80, 80개 부류의 인식률
            class_id=np.argmax(scores)  # 가장 큰 인식률의 부류
            confidence=scores[class_id] # 가장 큰 인식률
            if confidence>0.5:	# 신뢰도가 50% 이상인 경우만 취함
                #print(vec85)    # <2>
                centerx,centery=int(vec85[0]*width),int(vec85[1]*height)    # [0~1]표현을 이미지 내 실제 객체 중심위치로 변환
                w,h=int(vec85[2]*width),int(vec85[3]*height)                # [0~1]표현을 이미지 내 실제 객체 크기로 변환
                x,y=int(centerx-w/2),int(centery-h/2)                       # 객체의 좌측 상단 위치
                box.append([x,y,x+w,y+h])
                conf.append(float(confidence))
                id.append(class_id)
            
    ind=cv2.dnn.NMSBoxes(box,conf,0.5,0.4) # 비최대억제(NonMaximum Suppression) 알고리즘을 적용  # <3> 0.4 -> 1.0
    # score_threshold : a threshold used to filter boxes by score.
    # nms_threshold	: a threshold used in non maximum suppression. 40% 이상 겹치면 overlap->NMS 확인, 1.0이면 모든 box 출력
    objects=[box[i]+[conf[i]]+[id[i]] for i in range(len(box)) if i in ind]
    return objects

model,out_layers,class_names=construct_yolo_v3()		    # YOLO 모델 생성
colors=np.random.uniform(0,255,size=(len(class_names),3))	# 부류마다 색깔 다르게 설정 (랜덤으로)

img=cv2.imread('london_street.png') #eagle.jpg') #soccer.jpg')
if img is None:
    sys.exit('파일이 없습니다.')

res=yolo_detect(img,model,out_layers)	                    # YOLO 모델로 객체 검출
#print(len(res))

for i in range(len(res)):			# 검출된 물체를 영상에 표시
    x1,y1,x2,y2,confidence,id=res[i]
    text=str(class_names[id])+'%.3f'%confidence
    cv2.rectangle(img,(x1,y1),(x2,y2),colors[id],2)
    cv2.putText(img,text,(x1,y1+30),cv2.FONT_HERSHEY_PLAIN,1.5,colors[id],2)

cv2.imshow("Object detection by YOLO v.3",img)

cv2.waitKey()
cv2.destroyAllWindows()