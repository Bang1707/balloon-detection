import cv2
import numpy as np

cam = cv2.VideoCapture(0) #opening the default camera the one and only cam if we had another one 'd be(1)

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) #defining default frame width and height from pc

while True:
    ret, frame = cam.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Using cvtColor() func to swap the frame on the cam from BGR to HSV to implement further

    lower_blue = np.array([100, 150, 50]) #lower bound for inRange() function using numpy
    upper_blue = np.array([130, 255, 255]) # upper bound for inRange() function using numpy
    blue = cv2.inRange(img, lower_blue, upper_blue) #implementing inRange() function to binary mask blue colors

    contours, hierarchy = cv2.findContours(blue, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area = [cv2.contourArea(i) for i in contours]
    if area:
        if max(area) > 2000:
            cnt = contours[area.index(max(area))]
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255,0), 2)
            cv2.putText(frame, "Balloon detected", (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0 , 0), 3)

    

    cv2.imshow("Denis", frame)
    print(f"{area}")

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
