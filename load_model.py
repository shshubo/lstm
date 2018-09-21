# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import jieba.posseg as pseg
from tensorflow import keras
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

def encodeWord():
    vocabDict = {}
    vocabList = []
    vocabFile = open('temp/vocab.txt', 'r', encoding='utf-8')
    
    line = vocabFile.readline()
    while line:
        line = line.replace('\n', '')
        vocabList.append(line)
        line = vocabFile.readline()

    i = 1
    for word in vocabList:
        vocabDict[word] = i
        i = i + 1
    vocabFile.close()
    return vocabDict

def encodeText(filePath, vocabDict):
    resultList = [];
    resultLabelList =[];
    file = open(filePath, 'r', encoding='utf-8')
    line = file.readline()
    while line:
        line = line.replace('\n', '')
        tempWordList = []
        tempFlagList = []
        for word,flag in pseg.cut(line):
            if word in vocabDict:
                tempWordList.append(vocabDict[word])
                if flag == 'n':
                    tempFlagList.append(1)
                elif flag == 'nr':
                    tempFlagList.append(2)
                elif flag == 'ns':
                    tempFlagList.append(3)
                elif flag == 'nt':
                    tempFlagList.append(4)
                elif flag == 'nx':
                    tempFlagList.append(5)
                elif flag == 'nz':
                    tempFlagList.append(6)
                elif flag == 't':
                    tempFlagList.append(7)
                elif flag == 'w':
                    tempFlagList.append(8)
                else:
                    tempFlagList.append(9)
            else:
                tempWordList.append(0)
                tempFlagList.append(0)
        resultList.append(tempWordList);
        resultLabelList.append(tempFlagList);
        line = file.readline()
    resultList = pad_sequences(resultList, maxlen=100, padding='post')
    resultLabelList = pad_sequences(resultLabelList, maxlen=100, padding='post')
    file.close()
    return resultList, resultLabelList


vocabDict = encodeWord()
model = keras.Sequential()
model.add(keras.layers.Embedding(len(vocabDict), 32, input_length=100))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True), merge_mode='sum'))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(10, activation=tf.nn.softmax)))

model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights('save/cp-0030.ckpt')

testList,testLabelList = encodeText('data/sample.txt', vocabDict)
test = np.array(testList)
test_label = to_categorical(np.array(testLabelList))

loss, acc = model.evaluate(test, test_label)
print(loss)
print(acc)

result = model.predict(test)
for i in range(0,79):
    for j in range(0,30):
        print(np.argmax(result[i][j]))
        print(testLabelList[i][j])
        print("=========================================")
    print("+++++++++++++++++++++++++++++++++++++++++")