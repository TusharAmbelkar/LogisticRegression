
import numpy as np
import logging
import json
from utility import * #custom methods for data cleaning

FILE_NAME_TRAIN = 'train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'test.csv' #replace
ALPHA = 0.005
EPOCHS = 10000#keep this greater than or equl to 5000 strictly otherwise you will get an error
MODEL_FILE = 'models/model2'
train_flag = True

logging.basicConfig(filename='output.log',level=logging.DEBUG)

np.set_printoptions(suppress=True)
#################################################################################################
#####################################write the functions here####################################
#################################################################################################
#this function appends 1 to the start of the input X and returns the new array
def appendIntercept(X):
    #steps
    #make a column vector of ones
    #stack this column vector infront of the main X vector using hstack
    #return the new matrix
    #pass#remove this line once you finish writing
    rows,column=X.shape
    return np.hstack((np.ones((rows,1)),X))



 #intitial guess of parameters (intialize all to zero)
 #this func takes the number of parameters that is to be fitted and returns a vector of zeros
def initialGuess(n_thetas):
    return np.zeros((1,n_thetas))


def train(theta, X, y, model):
     J = [] #this array should contain the cost for every iteration so that you can visualize it later when you plot it vs the ith iteration
     #train for the number of epochs you have defined
     m = len(y)
     #your  gradient descent code goes here
     #steps
     #run you gd loop for EPOCHS that you have defined
        #calculate the predicted y using your current value of theta
        # calculate cost with that current theta using the costFunc function
        #append the above cost in J
        #calculate your gradients values using calcGradients function
        # update the theta using makeGradientUpdate function (don't make a new variable assign it back to theta that you received)
     m,n=X.shape
     for x in range(0,EPOCHS):
        y_predicted=predict(X,theta)
        #error=costFunc(m,y,y_predicted)
        #J.append(error)
        grads=calcGradients(X,y,y_predicted,m)
        theta=makeGradientUpdate(theta,grads)
     #print J

    # model['J'] = J
     model['theta'] = list(theta.flatten())
     #print model['theta']
     return model


#this function will calculate the total cost and will return it
def costFunc(m,y,y_predicted):
    #takes three parameter as the input m(#training examples), (labeled y), (predicted y)
    #steps
    #apply the formula learnt
    #pass
    return -(y*np.log(y_predicted)+(1-y)*np.log(1- y_predicted))

def calcGradients(X,y,y_predicted,m):
    #apply the formula , this function will return cost with respect to the gradients
    # basically an numpy array containing n_params
   #pass
   #print np.sum(np.multiply(X,(y_predicted-y)))/(m)
   rows,columns=X.shape
   #arr=np.repeat()
   #return np.sum()
   #print y_predicted
   #print len(y)
   #print y.shape

   return np.sum(np.multiply((y_predicted.T.reshape(m,1)-y.reshape(m,1)),X),axis=0)/m
  # return np.sum(np.multiply(X,(y_predicted[0]-y).reshape(m,1)),axis=0)/(m)

#this function will update the theta and return it
def makeGradientUpdate(theta, grads):
    #pass

    return theta-ALPHA*grads


#this function will take two paramets as the input
def predict(X,theta):
    #pass
    h=1.0/(1.0+np.exp(-np.dot(X,theta.T)))
    y_predicted=[]
    for i in range(0,len(X)):
        if h[i] < 0.5:
            y_predicted.append(0)
        else:
            y_predicted.append(1)
    #print np.array(y_predicted).shape
    return np.array(y_predicted).reshape(len(X),1)


########################main function###########################################
def main():
    if(train_flag):
        model = {}
        X_df,y_df = loadData(FILE_NAME_TRAIN)
        X,y, model = normalizeData(X_df, y_df, model)
        y=y_df
        X = appendIntercept(X)
        theta = initialGuess(X.shape[1])
        model = train(theta, X, y, model)
        with open(MODEL_FILE,'w') as f:
            f.write(json.dumps(model))

    #else:
        model = {}
        with open(MODEL_FILE,'r') as f:
            model = json.loads(f.read())
            X_df, y_df = loadData(FILE_NAME_TEST)
            X,y = normalizeTestData(X_df, y_df, model)
            y=y_df
            X = appendIntercept(X)
            accuracy(X,y,model)

if __name__ == '__main__':
    main()
