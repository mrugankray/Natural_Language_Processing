# importing all libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#importing dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',quoting = 3, delimiter = '\t')

# cleaning the texts
import re 
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(words) for words in review if not words in  set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
# building bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#splitting the dataset
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#fitting the calissifier into the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#predicting test set results
y_pred = classifier.predict(X_test)
 
#creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#accuracy
accuracy = ((55+91)/200)*100

#pickling
outfile = open("y_pred_new.pickle","wb")
pickle.dump(y_pred,outfile)
outfile.close()

#unpicling
pickle_in = open("y_pred_new.pickle","rb")
saved_y_Pred = pickle.load(pickle_in)
print(saved_y_Pred)