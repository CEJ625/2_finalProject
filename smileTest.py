import cv2  #영상을 처리하기 위해 필요한 라이브러리
import numpy as np   #벡터, 행렬 등 수치 연산을 수행하는 선형대수(Linear algebra) 라이브러리
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# Face detection XML load and trained model loading
face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#캐스케이드 분류기, 얼굴 검출, haarcascade_frontalface_default.xml= 얼굴인식 학습결과 파일
emotion_classifier = load_model('emotion_model.hdf5', compile=False)
#h5 또는 hdf5(표정인식 학습결과 파일)로 저장된 모델 구조를 불러옴
EMOTIONS = ["Angry" ,"Disgusting","Fearful", "Happy", "Sad", "Surpring", "Neutral"] #화남,  역겨운, 두려운,  행복, 슬픈, 놀란, 무표정

# 웹캠을 이용하여 비디오 재생, 카메라 열기 
camera = cv2.VideoCapture(0)  

while True:
    # 재생되는 비디오를 한 프레임씩 읽는다, 정상적인 값을 받으면, ret가 True이고, camera.read하면 한 프레임씩 읽어들임
    ret, frame = camera.read()
    
    # 이미지(배경)를 회색조로 변경 (배경과 현재 입력 프레임(frame)과의 차영상을 이용하여 객체를 검출)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image, 얼굴 검출 
    faces = face_detection.detectMultiScale(gray,  #대상 이미지 
                                            scaleFactor=1.1, # 이미지 스케일 
                                            minNeighbors=5,  # 얼굴 검출 후보들의 개수 
                                            minSize=(30,30)) # 가능한 최소 객체(얼굴) 사이즈 
    
    # 감정 예측 데이터 값을 나타낼 검정색 캔버스 생성 
    canvas = np.zeros((250, 300, 3), dtype="uint8") #zeros : numpy의 배열을 생성해주는 함수, 원소를 0로 배열을 생성함
    
    if len(faces) > 0:  #이미지를 검출했을 때 
        # For the largest image, 검출한 이미지를 face에 정렬하여 넣음
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face
        # Resize the image to 48x48 for neural network, 이미지 픽셀 조정 부분 ?
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # Emotion predict, 감정 예측 
        preds = emotion_classifier.predict(roi)[0]
        #표정 인식학습 결과 파일로 roi를 예측하여 값을 preds에 넣는다 (숫자형태)
        emotion_probability = np.max(preds)
        #np에서 가장큰  예측값을 가져와  emotion_probability에 넣음
        label = EMOTIONS[preds.argmax()]
        #예측한 값 중 가장 큰 값을 가져와 EMOTIONS배열의 인덱스 값을 라벨에 넣는다
        #예측한 값 중 가장 큰 값이 happy이면 라벨에 happy를 넣음 
        
        # Assign labeling, 프레임(얼굴 영상 창)에서 사각형 위에 검출한 표정 라벨 출력 
        cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
 
        # Label printing, 캔버스(표정 검출 데이터 창)에 라벨링
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            #zip : EMOTIONS리스트와 preds리스트의 값이 순서 쌍으로 묶음 ex)[(happy(EMOTIONS), 예측 값(preds))]
            #enumerate : 순서쌍을 열거
            #순서쌍 데이터에서 emotion=EMOTION, preds=preds 요소와 인덱스를 꺼내옴 
            text = "{}: {:.2f}%".format(emotion, prob * 100) #출력결과 -> happy: 35.05% 백분율 
            w = int(prob * 300)
            #표정 검출 데이터 창에서 예측된 값만큼 빨간 사각형으로 시각화하는 부분 
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            #표정 검출 데이터 창에서 예측된 값을 text로 출력하는 부분 
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)      

    label = EMOTIONS[preds.argmax()]
    test = prob * 100
    if(test >= 80):
        print(label) #80넘는 측정 값만 나옴 
        
    # Open two windows
    ## Display image ("Emotion Recognition")
    ## Display probabilities of emotion
    cv2.imshow('Emotion Recognition', frame)
    cv2.imshow("Probabilities", canvas)
    
    # q를 누르면 나가짐 
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q') or k == 27: #q또는 esc키 누를시 나감 
        break
    elif k == ord('c'):  #preds != None은 사용 못할 듯 무조건 예측치 중 큰 값이 있기 때문에 
        percent = max(preds)    # 표정 예측치 중 가장 큰 값
        index = np.where(percent == preds)[0][0]    # 가장 큰 표정 예측치의 index 값 구하기
        print(EMOTIONS[index], percent*100)
        continue

# Clear program and close windows
camera.release()
cv2.destroyAllWindows()
