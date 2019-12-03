import pandas as pd
data = pd.read_csv('6-Texts.csv')
msg =data['msg']
labels = data['labels']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(msg,labels)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
xtrain_dtm = cv.fit_transform(xtrain)
xtest_dtm = cv.transform(xtest)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(xtrain_dtm,ytrain)
pred = model.predict(xtest_dtm)
for doc,p in zip(xtest,pred):
    print(doc,'-->',p)
    
from sklearn import metrics
print(metrics.accuracy_score(pred,ytest))
print(metrics.confusion_matrix(pred,ytest))
print(metrics.recall_score(pred,ytest))
print(metrics.precision_score(pred,ytest))
