import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mtp
import copy

im_row,im_col=28,28
mnist=tf.keras.datasets.mnist

(x_train, y_train),(x_test,y_test)=mnist.load_data()
demo=copy.deepcopy(x_test)


x_train=tf.keras.utils.normalize(x_train,1)
x_test=tf.keras.utils.normalize(x_test,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
#
#
if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, im_row, im_col)
    x_test = x_test.reshape(x_test.shape[0], 1, im_row, im_col)
    input_shape = (1, im_row, im_col)
else:
    x_train = x_train.reshape(x_train.shape[0], im_row, im_col, 1)
    x_test = x_test.reshape(x_test.shape[0], im_row, im_col, 1)
    input_shape = (im_row, im_col, 1)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,activation="relu",input_shape=[28*28]))
model.add(tf.keras.layers.Dense(64,activation="relu",input_shape=[28*28]))
model.add(tf.keras.layers.Dense(10,tf.nn.softmax))


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="categorical_crossentropy",
              metrics=["accuracy"],
              )

model.fit(x_train,y_train,128,5,validation_split=0.2)
score = model.evaluate(x_test, y_test, verbose=0)
pred=model.predict(x_test)
print(score)
for i in range(50):

    # print(pred)
    #
    print("prediction: ",np.argmax(pred[i]))
    print()
    print("answer: ",np.argmax(y_test[i]))
    #
    mtp.imshow(demo[i])
    mtp.show()
