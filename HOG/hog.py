import cv2


img = cv2.imread('./img/person.png')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}
human, r = hog.detectMultiScale(img, **hogParams)
for(x, y, w, h) in human:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 50, 255), 3)

cv2.imwrite('./HOG/hog.png', img)
