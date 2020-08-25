import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
import numpy as np

#creating camera instance
cam = cv2.VideoCapture(0)

#creating the hog descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    ret, image = cam.read()

    #preprocessing the image
    image = imutils.resize(image, width=min(400, image.shape[1]))

    #detection
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)

    #applying non-max-suppression
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    #creating the non max suppression rectangles
    for (xA, yA, xB, yB) in pick:
	    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    #displaying image
    cv2.imshow("Detected pedestrians !", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()