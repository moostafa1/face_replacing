import cv2
import numpy as np
from joining_images import stackImages

img = cv2.imread(r'images (3).jpeg')
mask = cv2.imread(r'images (4).jpeg')


img = cv2.resize(img, (653, 470))
mask = cv2.resize(mask, (653, 470))

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
maskGray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces_rect = haar_cascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=3)
faces_mask = haar_cascade.detectMultiScale(maskGray, scaleFactor=1.1, minNeighbors=3)


print(f"Number of faces found = {len(faces_rect)}")
print(f"faces found = {faces_rect}")    # [[279  36 106 106]]



blank = np.zeros((img.shape[:2]), dtype='uint8')
# cv2.imshow('Blank Image', blank)




for (x,y,w,h) in faces_mask:
    axis_w = int(np.sqrt(w**2 + h**2))//4
    axis_h = int(np.sqrt(w**2 + h**2))//3

    face_mask = cv2.ellipse(blank.copy(), (x+w//2-3, y+h//2-3), (axis_w, axis_h), 0, 0, 360, (255,255,255), -1)
    masked = cv2.bitwise_and(mask, mask, mask=face_mask)

    cv2.ellipse(mask, (x+w//2-3, y+h//2-3), (axis_w, axis_h), 0, 0, 360, (0,255,0), 2)

print("mask ellipse center: ", (x+w//2-3, y+h//2-3))     # (219, 92)



for (x,y,w,h) in faces_rect:
    axis_w = int(np.sqrt(w**2 + h**2))//4
    axis_h = int(np.sqrt(w**2 + h**2))//3

    without_face = cv2.ellipse(img.copy(), (x+w//2-3, y+h//2-3), (axis_w, axis_h), 0, 0, 360, (0,0,0), -1)
    cv2.ellipse(img, (x+w//2, y+h//2), (axis_w, axis_h), 0, 0, 360, (0,255,0), 2)

print("img ellipse center: ", (x+w//2-3, y+h//2-3))    # (329, 86)




def translation(img, x, y):
    transMat = np.float32([[1,0,x], [0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv2.warpAffine(img, transMat, dimensions)

translated = translation(masked, 110, -7)
#cv2.imshow("translated face mask", translated)


"""

def rescaleFrame(frame, scale=0.95):
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale )

    dimensions = (width,height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


rescaled = rescaleFrame(translated, 0.95)

"""

final_mask = cv2.bitwise_or(without_face, translated)

imgStack = stackImages(0.5,([img, mask], [without_face, masked], [translated, final_mask]))

#cv2.imshow("Final Image", final_mask)
#cv2.imshow("Small Ramy face", masked)
#cv2.imshow("Big Ramy without face", without_face)
#cv2.imshow("Big Ramy", img)
#cv2.imshow("Small Ramy", mask)
cv2.imshow("output", imgStack)

cv2.waitKey(0)
