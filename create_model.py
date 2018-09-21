# -*- coding: UTF-8 -*-

import os
import jieba
import numpy as np
import tensorflow as tf
import jieba.posseg as pseg
from tensorflow import keras
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# np.set_printoptions(threshold=np.inf)

def createWordList():
    vocabSet = set()
    jieba.load_userdict("data/jieba_dic.txt")
    trainFile = open('data/train.txt', 'r', encoding='utf-8')
    testFile = open('data/test.txt', 'r', encoding='utf-8')
    vocabFile = open('temp/vocab.txt', 'w', encoding='utf-8')
    
    line = trainFile.readline()
    while line:
        for word in jieba.cut(line):
            if word.strip() != '':
                vocabSet.add(word)
        line = trainFile.readline()
    
    line = testFile.readline()
    while line:
        for word in jieba.cut(line):
            if word.strip() != '':
                vocabSet.add(word)
        line = testFile.readline()
    
    for word in vocabSet:
        vocabFile.write(word + '\n')
    
    vocabFile.close()
    testFile.close()
    trainFile.close()

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

def data_preProcess():
    vocabDict = encodeWord();
    
    trainList, trainLabelList = encodeText('data/train.txt', vocabDict)
    train = np.array(trainList)
    train_label = np.array(trainLabelList)
    
    valList, valLabelList = encodeText('data/test.txt', vocabDict)
    val = np.array(valList)
    val_label = np.array(valLabelList)

    tempTrain = open('temp/temp_train.txt', 'w', encoding='utf-8')
    tempTrainLabel = open('temp/temp_train_label.txt', 'w', encoding='utf-8')

    tempVal = open('temp/temp_val.txt', 'w', encoding='utf-8')
    tempValLabel = open('temp/temp_val_label.txt', 'w', encoding='utf-8')

    for seq in train:
        for word in seq:
            tempTrain.write(str(word) + '//')
        tempTrain.write('\n')
    for seq in train_label:
        for label in seq:
            tempTrainLabel.write(str(label) + '//')
        tempTrainLabel.write('\n')

    for seq in val:
        for word in seq:
            tempVal.write(str(word) + '//')
        tempVal.write('\n')
    for seq in val_label:
        for label in seq:
            tempValLabel.write(str(label) + '//')
        tempValLabel.write('\n')    

    tempTrain.close()
    tempTrainLabel.close()
    tempVal.close()
    tempValLabel.close()

def get_tempData(filePath):
    resultList = []
    file = open(filePath, 'r', encoding='utf-8')
    line = file.readline()
    while line:
        line = line.replace('\n', '')
        lineList = [x for x in line.split('//')]
        lineList.pop(-1)
        resultList.append(lineList)
        line =file.readline()
    return resultList

def plot_history(histories, key='loss'):
  plt.figure(figsize=(16,10))
    
  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])
  plt.show()

if __name__ == '__main__':

    # createWordList()
    # data_preProcess()

    vocabDict = encodeWord();

    train = np.array(get_tempData('temp/temp_train.txt'))
    train_label = np.array(get_tempData('temp/temp_train_label.txt'))
    train_label = to_categorical(train_label, 10)

    val = np.array(get_tempData('temp/temp_val.txt'))
    val_label = np.array(get_tempData('temp/temp_val_label.txt'))
    val_label = to_categorical(val_label, 10)


    model = keras.Sequential()
    model.add(keras.layers.Embedding(len(vocabDict), 32, input_length=100))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True), merge_mode='sum'))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(10, activation=tf.nn.softmax)))
    
    model.summary()
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint_path = "save/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cpCallback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     period=10)

    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

    history = model.fit(train, train_label,
                validation_data = (val, val_label),
                epochs=30,
                batch_size=1000,
                callbacks=[cpCallback, tbCallBack],
                verbose=1)

    # testList,testLabelList = encodeText('data/sample.txt', vocabDict)
    # test = np.array(testList)
    # test_label = to_categorical(np.array(testLabelList))

    # loss, acc = model.evaluate(test, test_label)
    # print(loss)
    # print(acc)

    # result = model.predict(test)
    # for i in range(0,79):
    #     for j in range(0,30):
    #         print(np.argmax(result[i][j]))
    #         print(testLabelList[i][j])
    #         print("=========================================")
    #     print("+++++++++++++++++++++++++++++++++++++++++")

    plot_history([('baseline', history)])