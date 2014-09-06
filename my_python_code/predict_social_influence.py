# written by Chan Yi Sheng
 
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import auc_score
from sklearn import cross_validation as cv
import numpy as np
 
###########################
# LOADING TRAINING DATA & MAKE CROSS VALIDATION FOLDS
###########################
 
trainfile = open('train.csv')
header = trainfile.next().rstrip().split(',')
 
y_train = []
X_train_A = []
X_train_B = []
 
for line in trainfile:
    splitted = line.rstrip().split(',')
    label = int(splitted[0])
    A_features = [float(item) for item in splitted[1:12]]
    B_features = [float(item) for item in splitted[12:]]
    y_train.append(label)
    X_train_A.append(A_features)
    X_train_B.append(B_features)
trainfile.close()
 
y_train = np.array(y_train)
X_train_A = np.array(X_train_A)
X_train_B = np.array(X_train_B)
 
###########################
# Use log transform of the differences and in order to avoid log of zero
# I perfrom a little trick here
###########################
 
def transform_features(x):
    return np.log(x)
 
for point in X_train_A:
    for i in range(11):
        point[i] = point[i]+1
    
for point in X_train_B:
    for i in range(11):
        point[i] = point[i]+1


X_train = transform_features(X_train_A) - transform_features(X_train_B)
model = linear_model.LogisticRegression(fit_intercept=False)
model.fit(X_train,y_train)

## compute AuC score on the training data using Logistic Regression / Random Forest ##

# Logistic Regression
p_train = model.predict_proba(X_train)
p_train = p_train[:,1:2]
auc = auc_score(y_train,p_train)
print('AUC score = ', round(auc,3),'Using Logistic Regression' ) 
scores_lr = cv.cross_val_score(model, X_train,y_train,cv=5).mean()

# Random forest
modelrf = rf(n_estimators=300,max_depth=6,max_features='auto',oob_score=True).fit(X_train,y_train)
p_train2 = modelrf.predict_proba(X_train)
p_train2 = p_train2[:,1:2]
auc2 = auc_score(y_train,p_train2)
print('AUC score = ', round(auc2,5),'Using Random Forest' ) 
scores_rf = cv.cross_val_score(modelrf, X_train,y_train,cv=5).mean()


###########################
# LOADING TEST DATA
###########################
 
#ignore the test header
testfile = open('test.csv')
testfile.next()
 
X_test_A = []
X_test_B = []
for line in testfile:
    splitted = line.rstrip().split(',')
    A_features = [float(item) for item in splitted[0:11]]
    B_features = [float(item) for item in splitted[11:]]
    X_test_A.append(A_features)
    X_test_B.append(B_features)
testfile.close()
  
X_test_A = np.array(X_test_A)
X_test_B = np.array(X_test_B)
 
for point in X_test_A:
    for i in range(11):
        point[i] = point[i]+1
    
for point in X_test_B:
    for i in range(11):
        point[i] = point[i]+1
 
 
# transform features
X_test = transform_features(X_test_A) - transform_features(X_test_B)
# compute probabilistic predictions under different models
p_test_lr = model.predict_proba(X_test)
p_test_lr = p_test_lr[:,1:2]
p_test_rf = modelrf.predict_proba(X_test)
p_test_rf = p_test_rf[:,1:2]

 
###########################
# WRITING SUBMISSION FILE to KAGGLE, USING BETTER MODEL WHICH IS RANDOM FOREST
###########################
predfile = open('predictions.csv','w+')
 
print >>predfile,','.join(header)
for line in np.concatenate((p_test_rf,X_test_A,X_test_B),axis=1):
    print >>predfile, ','.join([str(item) for item in line])
 
predfile.close()