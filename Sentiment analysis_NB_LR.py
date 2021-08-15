#Sentiment analysis of tweets using logistic regression and a comparison with a Naive Bayes model
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report

#defining model for the sentiment analysis
def sentiment_model(model_name, model, X_train, X_test, y_train, y_test):
    print(f'BEGIN. {model_name.upper()}......')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    print(f'TESTING DATA----> {model_name.upper()}: \t\t{accuracy_score(y_test, y_pred) * 100:.2f}%') #accuracy of model on testing data
    print(f'TRAINING DATA---> {model_name.upper()}: \t\t{accuracy_score(y_train, y_train_pred) * 100:.2f}%') #accuracy of model on training data
    print(classification_report(y_test, y_pred))
    print(f'END. {model_name.upper()}')
    print('======================================================')
    return y_pred

import nltk
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
lemmatizer = PorterStemmer()
stop = stopwords.words('english')
import re

#preprocess the text
def preprocess(text, lemmatize=False):
    # Remove url, change to lower case
    text = re.sub('http\S+|www\S+', ' ', str(text).lower())
    # remove picture links
    text = re.sub('pic\S*\s?', ' ', text)
    # remove all handles
    text = re.sub('@\S+', ' ', text)
    # remove special characters
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    # remove all multiple white spaces
    text = re.sub('[\s]+', ' ', text).strip()
    #remove all stop words
    text = ' '.join([word for word in text.split() if word not in (stop)])
    #tokenize text, strip suffixes and return joined text
    if lemmatize:
        word_list = nltk.word_tokenize(text)
        lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
        return lemmatized_output  # " ".join(tokens)
    else:
        return text

#import and read the train and test datasets
#store the datasets in variables
data = pd.read_csv("Corona_NLP_train.csv", encoding='latin-1')#train data
data_t = pd.read_csv("Corona_NLP_test.csv", encoding='latin-1')#test data

#extracting the independent variable, i.e., tweets and the dependent variable i.e., associated sentiment in the train data
X = data.OriginalTweet
y = data.Sentiment
#print number of rows and columns in train data
print('Shape of train data:')
print(f'Data shape: {data.shape}')

#extracting the independent variable, i.e., tweets and the dependent variable i.e., associated sentiment in the train data
x = data_t.OriginalTweet
Y = data_t.Sentiment
#print number of rows and columns in test
print('Shape of test data')
print(f'Data shape: {data_t.shape}')
#plot to show the distribution of sentiments
import matplotlib.pyplot as plt
plt.hist(data['Sentiment'],edgecolor='white',bins=5)
plt.show()
#for train data
#extract the tweets
df_train = data["OriginalTweet"]
#preprocess the tweets
res = preprocess(df_train)
#store the pre processed tweet in a list
alist = data["OriginalTweet"].tolist()
X_data = []
#append the words remaining after preprocessing in a list
for text in alist :
    X_data.append(preprocess(text))
#store the preprocessed train data as an array
X_d = np.array(X_data)
print('Preprocessed train data shape:')
print(X_d.shape, len(X_data), X.shape, y.shape)

#for test data
#extract the tweets
df_test=data_t["OriginalTweet"]
#preprocess the tweets
res_t=preprocess(df_test)
#store the pre processed tweet in a list
blist = data_t["OriginalTweet"].tolist()
x_data=[]
#append the words remaining after preprocessing in a list
for text in blist:
    x_data.append(preprocess(text))
#store preprocessed test data as array
x_d = np.array(x_data)
print('Preprocessed test data shape:')
print(x_d.shape, len(x_data), x.shape, Y.shape)

#encoding categorical data into numbers in order to form  the model equation
from sklearn.preprocessing import LabelEncoder
#label encoder class encodes the variables into digits
encoder = LabelEncoder()
encoder.fit(data['Sentiment'].tolist())
# emotions encoded as 0:Extremely negative, 1:Extremely positive, 2: negative, 3:neutral, 4:positive
y_train = encoder.transform(y.tolist())
y_test = encoder.transform(Y.tolist())
print("Shape of y_train",y_train.shape)
print("Shape of y_test",y_test.shape)
#printing the shape of dependent and independent variables in train and test data to ensure the dimensions match
print(f'X_Train shape: {X_d.shape}, y_train shape: {y_train.shape}')
print(f'X_Test shape: {x_d.shape}, y_test shape: {y_test.shape}')

print('\n============Message Preprocessing============')
#converting the text into a matrix of token counts
vectorizer = CountVectorizer(ngram_range=(1, 3), max_df=1.0, min_df=3, stop_words='english')
X_train_vect = vectorizer.fit_transform(X_d)
X_test_vect = vectorizer.transform(x_d)


print('Training and testing data shape after pre-processing:')
print(f'X_Train shape: {X_train_vect.shape}, y_train shape: {y_train.shape}') #print train data shape
print(f'X_Test shape: {X_test_vect.shape}, y_test shape: {y_test.shape}') #print test data shape

print('\n=============Model Building==================')
#logistic regression model
lr_model = LogisticRegression(max_iter=10000) #max_iter value set manually to allow model to converge
lr_y_pred = sentiment_model('Logistic Regression', lr_model, X_train_vect, X_test_vect, y_train, y_test)
#Naive bayes model
nb_model = MultinomialNB()
nb_y_pred = sentiment_model('Naive Bayes', nb_model, X_train_vect, X_test_vect, y_train, y_test)
