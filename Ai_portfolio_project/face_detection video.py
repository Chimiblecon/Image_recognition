import cv2
train_paste_data= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
web_cam = cv2.VideoCapture('babies.mp4')
while True:
    successful_frame_read, frame= web_cam.read()

    grayscale_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    face_cordinate =train_paste_data.detectMultiScale(grayscale_image)
    for (x,y,w,h) in face_cordinate:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)
    cv2.imshow('video_window', frame)
    cv2.waitKey(1)

print('code executed successfully')
