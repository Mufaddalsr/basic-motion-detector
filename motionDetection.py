import cv2
import pandas as pd
from datetime import datetime


# @author    Mufaddal Ragib <ragibmufaddal@gmail.com>
# @see      https://github.com/Mufaddalsr/basic-motion-detector/ The motion detector GitHub project

# @note      This program is distributed in the hope that it will be useful - WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.


firstFrame = None
statusList = [None, None]
times = []
df = pd.DataFrame(columns=['Start Time', 'End Time'])

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    status = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gray
        continue

    deltaFrame = cv2.absdiff(firstFrame, gray)
    threshFrame = cv2.threshold(deltaFrame, 30, 255, cv2.THRESH_BINARY)[1]
    threshFrame = cv2.dilate(threshFrame, None, iterations=2)

    (cnts, _) = cv2.findContours(threshFrame.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue

        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    statusList.append(status)
    if statusList[-1] == 1 and statusList[-2] == 0:
        times.append(datetime.now())
    if statusList[-1] == 0 and statusList[-2] == 1:
        times.append(datetime.now())
    cv2.imshow('Motion Detection', gray)
    cv2.imshow('delta frame', deltaFrame)
    cv2.imshow('thresh delta ', threshFrame)
    cv2.imshow('Color frame ', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

    print(status)
    
print(times)
for i in range(0, len(times), 2):
    df=df.append({'Start Time': times[i],
              'End Time': times[i+1]}, ignore_index=True)

df.to_csv('dataset/motionTimes.csv')

video.release()
cv2.destroyAllWindows()
