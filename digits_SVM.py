from sklearn import datasets
from sklearn import svm
import cv2

digits = datasets.load_digits()
clf = svm.SVC(gamma = 0.001, C = 100)
x,y = digits.data[:-1],digits.target[:-1]
clf.fit(x,y)

for i in range(len(y)):
    data = digits.data[i][:]
    print("Prediction: ", clf.predict(data.reshape(1, -1)))
    img = data.reshape(8, 8)
    image = cv2.resize(img, (320, 320), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('image', image)
    key = cv2.waitKey(0)
    if key & 0xff == ord('q'):
        continue
