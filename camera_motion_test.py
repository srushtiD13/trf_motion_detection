import cv2
import numpy as np
cap = cv2.VideoCapture('motion1.mp4')

_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

lk_params = dict(winSize = (15, 15),
                 maxLevel = 4,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


print(frame.shape)
while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for i in range(0, 600, 70):
        for j in range(0, 200, 70):
            old_points = np.array([[i, j]], dtype=np.float32)
            new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
            x1, y1 = new_points.ravel()
            if(abs(i-x1)>9 and abs(j-y1)>9):
                cv2.line(frame, (i, j), (x1, y1), (0, 255, 0), 4)

    print(new_points)
    old_gray = gray_frame.copy()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 32:
        break
cap.release()
cv2.destroyAllWindows()