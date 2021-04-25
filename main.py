import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

batch_size = 32
embedding_dim = 256
epochs = 0
max_tokens = 2048
sentence_length = 32
raw_train_x = []
raw_train_y = []
train_size = 0
all_story = ""
for i in range(len(os.listdir('./dataset'))):
    file = open('./dataset/story' + str(i) + '.txt', encoding='utf-8')
    story = file.read()
    file.close()
    all_story += story
    tmp_size = len(story) - sentence_length
    train_size += tmp_size
    for j in range(tmp_size):
        raw_train_x.append(list(story[j:j + sentence_length]))
        raw_train_y.append(story[j + sentence_length])
vectorize_layer = TextVectorization(max_tokens=max_tokens, output_mode='int', output_sequence_length=1)
vectorize_layer.adapt(list(all_story))


def vectorize_text(text, length):
    tmp_vector = vectorize_layer(text).numpy()
    vector = np.zeros(length)
    for v_i in range(length):
        if v_i < len(text):
            vector[length - v_i - 1] = tmp_vector[len(text) - v_i - 1][0]
        else:
            break
    return vector


train_x = np.zeros([train_size, sentence_length])
train_y = vectorize_text(raw_train_y, train_size)
for i in range(train_size):
    train_x[i] = vectorize_text(raw_train_x[i], sentence_length)
state = np.random.get_state()
np.random.shuffle(train_x)
np.random.set_state(state)
np.random.shuffle(train_y)
if 'model' in os.listdir('.'):
    model = tf.keras.models.load_model('./model')
else:
    model = tf.keras.Sequential([
        layers.Embedding(len(vectorize_layer.get_vocabulary()), embedding_dim),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(vectorize_layer.get_vocabulary()), activation='softmax')
    ])
model.compile(loss=losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)
model.save('./model')
topic = input()
sentence = np.zeros([1, sentence_length])
sentence[0] = vectorize_text(list(topic), sentence_length)
while True:
    for k in range(100):
        pred = model.predict(sentence)[0]
        pred_index = np.argmax(pred)
        print(vectorize_layer.get_vocabulary()[pred_index], end='')
        for i in range(sentence_length - 1):
            sentence[0][i] = sentence[0][i + 1]
        sentence[0][sentence_length - 1] = pred_index
    input()
