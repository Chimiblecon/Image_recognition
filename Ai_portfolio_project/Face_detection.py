import cv2

#loading pretrained data
train_face_data= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#chosing image to detect
img= cv2.imread('rev_sisters.jpg')
#cv2.imshow('test_image',img)
#cv2.waitKey()
#converting picture to black_and_white
grayscale_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('converted',grayscale_img)
#cv2.waitKey()
#drawing_rectangles_2_detect_faces
face_cordinate= train_face_data.detectMultiScale(img)
print(face_cordinate)
for (x,y,w,h) in face_cordinate:
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 100, 126), 5)

cv2.imshow('face_detected',img)
cv2.waitKey()











print('program executed successfully')

