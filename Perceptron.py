from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
iris=load_iris()
X = iris['data'][:,0:2][:100]
y = iris['target'][:100]
y= np.array([1 if i==1 else -1 for i in y])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
'''
plt.scatter(X[:,0],X[:,1])
plt.show()
'''
w=np.ones(2,dtype=np.float32)
b=0
lr=0.005
flag=True
def train(X_train,y_train,w,b):
    flag= True
    while flag:
        wrong = 0
        for i in range(len(X_train)):
            X=X_train[i]
            y=y_train[i]
            y_pred = X@w + b
            if(y*y_pred<=0):
                wrong+=1
                w = w + lr * np.dot(y,X)
                b = b + lr * y
        if wrong==0:
                flag= False
        print('w={},b={}'.format(w, b))
    print('training completed')
    print('w={},b={}'.format(w,b))
    return w,b
def test(X_test,y_test,w,b):
    wrong = 0
    for i in range(len(X_test)):
        X = X_test[i]
        y = y_test[i]
        y_pred = X@w + b
        if(y*y_pred<=0):
            wrong+=1
    accuracy = 1 - wrong/len(X_test)
    print('testing completed')
    print('accuracy={}'.format(accuracy))
w,b=train(X_train,y_train,w,b)
test(X_test,y_test,w,b)
axis_a = np.linspace(4,7,100)

axis_b = -(w[0]*axis_a+b)/w[1]
blue_index = np.where(y_test==1)
red_index = np.where(y_test==-1)
blue = X_test[blue_index]
red = X_test[red_index]
plt.scatter(blue[:,0],blue[:,1],c='blue')
plt.scatter(red[:,0],red[:,1],c='red')
plt.legend(['1','0'])
plt.plot(axis_a,axis_b)
plt.show()
