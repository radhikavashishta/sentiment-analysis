# Sentiment analysis of tweets pertaining to Covid-19
Sentiment analysis (also known as opinion mining or emotion AI) refers to the use of natural language processing and text analysis to systematically identify, extract, quantify, and study affective states and subjective information. A basic task in sentiment analysis is classifying the polarity of a given text at the document, sentence, or feature/aspect level—whether the expressed opinion in a document, a sentence or an entity feature/aspect is positive, negative, or neutral. Not considering a neutral class can give rise to over-fitting and make the system vulnerable to cases where due to randomness a particular neutral word occurs more times in positive or negative examples.

# Model Details
The aim of the project is to classify tweets about the coronavirus and the covid-19 pandemic according to the sentiment expressed by them. A 5-level sentiment scale is adopted, with the sentiments being classified into the classes: Extremely Negative, Negative, Neutral, Positive and Extremely Positive. The [dataset](https://www.kaggle.com/immvab/transformers-covid-19-tweets-sentiment-analysis) used has been sourced from Kaggle. It is composed of tweets with covid-19 related hashtags that have been pulled from Twitter and have been manually tagged.
A model based on logistic regression was developed and trained on the data and tested to determine the accuracy, and a comparison was drawn against a model based on Naïve Bayes classification. A Naïve Bayes classifier belongs to the class of Generative classifiers which learn a model of the joint probability p(x,y) of the input x and the label y and make their predictions using Bayes rule to calculate p(y|x) and then picking the most likely label y. A Logistic Regression based classifier belongs to the class of Discriminative classifiers which, on the other hand, model p(y|x) directly, i.e. they learn a direct map from the input x to the label y. 

# Text Preprocessing
## Data cleaning: 
The process aims to remove irrelevant items, such as HTML tags, non-word characters etc. One of the most prominent tools used for this process is Regular Expressions. A regular expression or regex is a sequence of characters that define a search pattern.

## Normalization
Normalization converts text to all lowercase and removing punctuation.

## Sentence segmentation
Sentence tokenization or sentence segmentation is the process of breaking a string of text into its component sentences. Often punctuation marks can be used as a sentence boundary. However, abbreviations pose an issue since they use the period character after every alphabet, which can mess up sentence boundaries. A lookup table can help resolve this issue.

## Word Tokenization
Word tokenization or word segmentation involves dividing a string of written language into its component words. In many languages, the white space is a good approximation of a word boundary.

## Stop word filtering
Stop words are the most common words in a language such as articles, prepositions etc. They do not add much meaning to a sentence and add a lot of noise to the data being processed, so they need to be filtered out.

# Libraries Required
1. Numpy

2. Pandas

3. Scikit-learn

   Modules from scikit-learn: 

   a. CountVectorizer

   b. LogisticRegression

   c. MultinomialNB

   d. Accuracy_score

   e. Classification_report 

   f. LabelEncoder

4. Nltk

   Modules from nltk: 
   
   a. Punkt

   b. Stopwords

   c. PorterStemmer

