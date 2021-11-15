#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 # conda install -c conda-forge opencv
import numpy as np
import tensorflow as tf # 2.1.0
import pandas as pd # conda install -c anaconda pandas

img_rows, img_cols = 28, 28

if tf.keras.backend.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
    first_dim = 0
    second_dim = 1
else:
    input_shape = (img_rows, img_cols, 1)
    first_dim = 0
    second_dim = 3

# EMNIST balanced dataset
train = pd.read_csv("./emnist-balanced/emnist-balanced-train.csv",delimiter = ',')
test = pd.read_csv("./emnist-balanced/emnist-balanced-test.csv", delimiter = ',')
mapp = pd.read_csv("./emnist-balanced/emnist-balanced-mapping.txt", delimiter = ' ',                    index_col=0, header=None, squeeze=True)
print(f'Train: {train.shape}, Test: {test.shape}, Map: {mapp.shape}')


# In[2]:


X_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]
del train

X_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]
del test

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[3]:


def rotate(image):
    image = image.reshape([img_rows, img_cols])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image


# In[4]:


# Flip and rotate image
X_train = np.asarray(X_train)
X_train = np.apply_along_axis(rotate, 1, X_train)
print(f'X_train: {X_train.shape}')

X_test = np.asarray(X_test)
X_test = np.apply_along_axis(rotate, 1, X_test)
print(f'X_test: {X_test.shape}')


# In[5]:


# Normalize
X_train = X_train.astype('float32')
X_train = X_train / 255.0
X_test = X_test.astype('float32')
X_test = X_test / 255.0


# In[6]:


# number of classes
num_classes = y_train.nunique()

X_train = np.expand_dims(X_train, second_dim)
X_test = np.expand_dims(X_test, second_dim)


# In[7]:


# tensorflow.keras로 EMNIST 인식 모델 구축
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                            activation='relu',
                            input_shape=input_shape),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])


# In[8]:


# EMNIST 인식 모델 compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[9]:


# EMNIST 인식 모델 훈련
model.fit(X_train, y_train, epochs=5)


# In[10]:


# EMNIST 인식 모델 정보 출력
model.summary()


# In[11]:


# EMNIST 인식 모델 평가
model.evaluate(X_test, y_test) 


# In[12]:


model.save_weights('./emnist_cnn/emnist_cnn_checkpoint')


# In[13]:


font = cv2.FONT_HERSHEY_SIMPLEX

cp = cv2.VideoCapture(0)
cp.set(3, 5*128)
cp.set(4, 5*128)
SIZE = 28


# In[14]:


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


# In[15]:


print("loading model")
model.load_weights('./emnist_cnn/emnist_cnn_checkpoint')

labelz = dict(enumerate([chr(m) for m in mapp]))


# In[17]:


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




