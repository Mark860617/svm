from sklearn import svm
import cv2
from tensorflow.examples.tutorials.mnist import input_data


def onehot_to_list(input, ys_size):     # convert onehot label to a list
    labels = []
    for i in range(ys_size):
        for j in range(10):
            if input[i][j] == 1:
                labels.append(int(j))
    return labels


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_xs, batch_ys = mnist.train.next_batch(1000)
print("data read")
train_labels = onehot_to_list(batch_ys, len(batch_ys))
print("train label transformed")

clf = svm.SVC(gamma = 0.001, C = 100)
x,y = batch_xs,train_labels
clf.fit(x,y)
print("training completed!")

test_len = 100
test_data = mnist.test.images[:test_len]
test_label = mnist.test.labels[:test_len]
test_labels = onehot_to_list(test_label, len(test_label))
print("test labels transformed")

for i in range(len(test_labels)):
    data = test_data[i][:]
    print("Prediction: ", clf.predict(data.reshape(1, -1)))
    img = data.reshape(28, 28)
    image = cv2.resize(img, (320, 320), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('image', image)
    key = cv2.waitKey(0)
    if key & 0xff == ord('q'):
        continue

# # plt.imshow(digits.data[-1], cmap = plt.cm.gray_r, interpolation = 'nearest')
# # plt.show
