#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 # conda install -c conda-forge opencv
import numpy as np
import tensorflow as tf # 2.1.0

# MNIST dataset
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train.astype('float32')
X_train = X_train / 255.0
X_test = X_test.astype('float32')
X_test = X_test / 255.0


# In[3]:


# MNIST shape
img_rows, img_cols = 28, 28

if tf.keras.backend.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
    first_dim = 0
    second_dim = 1
else:
    input_shape = (img_rows, img_cols, 1)
    first_dim = 0
    second_dim = 3

X_train = np.expand_dims(X_train, second_dim)
X_test = np.expand_dims(X_test, second_dim)


# In[4]:


# tensorflow.keras로 MNIST 인식 모델 구축
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])


# In[5]:


# MNIST 인식 모델 compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[7]:


# MNIST 인식 모델 훈련
model.fit(X_train, y_train, epochs=5)


# In[8]:


# MNIST 인식 모델 정보 출력
model.summary()


# In[9]:


# MNIST 인식 모델 평가
model.evaluate(X_test, y_test)


# In[10]:


model.save_weights('./mnist_ffn/mnist_ffn_checkpoint')


# In[11]:


font = cv2.FONT_HERSHEY_SIMPLEX

cp = cv2.VideoCapture(0)
cp.set(3, 5*128)
cp.set(4, 5*128)
SIZE = 28


# In[12]:


def annotate(frame, label, location = (20,30)):
    #writes label on image#

    cv2.putText(frame, label, location, font,
                fontScale = 0.5,
                color = (255, 255, 0),
                thickness =  1,
                lineType =  cv2.LINE_AA)

def extract_digit(frame, rect, pad = 10):
    x, y, w, h = rect
    cropped_digit = final_img[y-pad:y+h+pad, x-pad:x+w+pad]
    cropped_digit = cropped_digit/255.0

    #only look at images that are somewhat big:
    if cropped_digit.shape[0] >= 32 and cropped_digit.shape[1] >= 32:
        cropped_digit = cv2.resize(cropped_digit, (SIZE, SIZE))
    else:
        return
    return cropped_digit

def img_to_mnist(frame, tresh = 90):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    #adaptive here does better with variable lighting:
    gray_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, blockSize = 321, C = 28)

    return gray_img


# In[13]:


print("loading model")
model.load_weights('./mnist_ffn/mnist_ffn_checkpoint')

# 숫자 라벨 list 생성
num_label_list = ["zero", "one", "two", "three", "four",
                         "five", "six", "seven", "eight", "nine"]
labelz = dict(enumerate(num_label_list))


# In[ ]:


for i in range(1000):
    ret, frame = cp.read(0)

    final_img = img_to_mnist(frame)
    image_shown = frame
    contours, _ = cv2.findContours(final_img.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(contour) for contour in contours]
    rects = [rect for rect in rects if rect[2] >= 3 and rect[3] >= 8]

    #draw rectangles and predict:
    for rect in rects:

        x, y, w, h = rect

        if i >= 0:

            mnist_frame = extract_digit(frame, rect, pad = 15)

            if mnist_frame is not None: #and i % 25 == 0:
                mnist_frame = np.expand_dims(mnist_frame, first_dim) #needed for keras
                mnist_frame = np.expand_dims(mnist_frame, second_dim) #needed for keras

                class_prediction = model.predict_classes(mnist_frame, verbose = False)[0]
                prediction = np.around(np.max(model.predict(mnist_frame, verbose = False)), 2)
                label = str(prediction) # if you want probabilities

                cv2.rectangle(image_shown, (x - 15, y - 15), (x + 15 + w, y + 15 + h),
                              color = (255, 255, 0))

                label = labelz[class_prediction]

                annotate(image_shown, label, location = (rect[0], rect[1]))

    cv2.imshow('frame', image_shown)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# In[ ]:




