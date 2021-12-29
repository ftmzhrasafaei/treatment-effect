import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import copy

def Confusion_Graph(cm, tit= False):

    ax = sns.heatmap(cm, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    if tit:
        plt.title(tit)
    ## Display the visualization of the Confusion Matrix.
    plt.show()



df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
train = df.sample(frac = 1)
validation = df.drop(train.index)
labels = ['Outcome' , 'Twoyears.follow_up.Cobb']
train_set = np.array(train.drop(labels , axis = 1))
train_label1 = np.array(train[labels[0]])
train_label2 = np.array(train[labels[1]])
validation_set = np.array(validation.drop(labels , axis = 1))
validation_label1 = np.array(validation[labels[0]])
validation_label2 = np.array(validation[labels[1]])


print('-----------------------------')

model1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(train_set.shape[1],)),  # input layer (1)
    keras.layers.Dense(150, activation='relu'), # hidden layer (2)
    keras.layers.Dense(2, activation='softmax') # output layer (3)
])

model1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


m1 = model1.fit(train_set, train_label1, epochs=10 , validation_split=0.1, shuffle=True)
plt.plot(m1.history['loss'])
plt.plot(m1.history['val_loss'])
plt.title('model loss of classifier')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


print('-----------------------------')
model2 = keras.Sequential([
    keras.layers.Dense(500, input_dim=train_set.shape[1], activation= "relu"),
    keras.layers.Dense(50, activation= "relu"),
    keras.layers.Dense(1)])

model2.compile(loss= "mse" , optimizer="rmsprop", metrics=["mse"])

m2 = model2.fit(train_set, train_label2, epochs=20, validation_split=0.1, shuffle=True)
plt.plot(m2.history['loss'])
plt.plot(m2.history['val_loss'])
plt.title('model loss of Regression')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


print('-----------------------------')

predictions = model1.predict(train_set)
p = [np.argmax(item) for item in predictions]
cm = confusion_matrix(train_label1 ,p )
Confusion_Graph(cm, 'Train dataset')


print('Do you want to evaluate the model with your dataset?')
while(True):
    print('Enter 1 for YES and 0 for NO')
    a = int(input())
    if a==0 or a == 1:
        break

if a:
    print('Enter the url of your dataset in xlsx format')
    url = input()
    test = pd.read_excel(url, sheet_name='Sheet1')
    labels = ['Outcome' , 'Twoyears.follow_up.Cobb']
    test_label1 = np.array(test[labels[0]])
    test_label2 = np.array(test[labels[1]])
    for col in test.columns:
        if col in labels:
            test = test.drop([col] , axis = 1)
    test = np.array(test)
    test_loss1, test_acc1 = model1.evaluate(test,  test_label1, verbose=1)

    print('Accuracy on test:', 100* test_acc1)
    predictionst = model1.predict(test)
    pt = [np.argmax(item) for item in predictionst]
    cmt = confusion_matrix(test_label1 ,pt )
    Confusion_Graph(cmt, 'On test')

    test_loss2, test_acc2 = model2.evaluate(test,  test_label2, verbose=1)

    print('MSE of regression on test:', test_acc2)
