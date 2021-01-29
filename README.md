# NLP-articles-Classificaton

### Problem statement

The AG's news topic classification dataset is constructed by choosing 4 largest classes from the original corpus. Each class contains 30,000 training samples and 1,900 testing samples. The total number of training samples is 120,000 and testing 7,600.
The file classes.txt contains a list of classes corresponding to each label.

The files train.csv and test.csv contain all the training samples as comma-sparated values. There are 3 columns in them, corresponding to class index (1 to 4), title and description. The title and description are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). New lines are escaped by a backslash followed with an "n" character, that is "\n".

This is a Classification Problem which implies you to apply an efficent algorithm that gives high accuracy on test Data.


### Approach

1) Import all the necessary libraries</br>
In this particular Solution NLTK , ScikitLearn , numpy , pandas are used.
```
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
```

2) Load the Train.csv and Test.csv file using pandas by omitting them headers and mapping the labels to respective classes. while Loading , you can replace the url with your own dataset path. 
``` python
df = pd.read_csv("your_csv_path" , header = None)
```

3) <b>Data Cleaning</b></br>
Before doing analysis on data we need to transform the dataset and clean it so that the computer is able to understand that data in the desired format.Data preprocessing is a fundamental step while building a machine learning model. If the data is fairly pre-processed the results would be reliable. In NLP, the first step before building the machine learning model, is to pre-process the data. In this particular Solution we have used various text cleaning methods like </br>
  --> stop words</br>
  --> lemmatization</br>
  --> stemming algorithm</br>
</br>
It is trade between speed and accuracy to Choose between stemming and lemmatization because in Stemming algorithm,it is used to extract the base form of the words by removing affixes from them. It is just like cutting down the branches of a tree to its stems. The stemmed word may or may not be in the context of dictionary.While in Lemmatization, words are stemmed to its root word which will be form the dictionary. Lemmatization takes more time in cleaning but accuracy is high and Stemming is purely opoosite to this, it gives priority to speed over accuracy.</br>
</br>

Here we have used Lemmatization for high accuracy.

4) Dividing the dataset into X_train, y_train, X_test, y_test.

5) <b>Word Vectorization</b></br>
Word vectorization is a methodology in NLP to map words or phrases from vocabulary to a corresponding vector of real numbers which used to find word predictions, word similarities/semantics.Here we have used TfidVectorizer() for word vectorization because In CountVectorizer we only count the number of times a word appears in the document which results in biasing in favour of most frequent words. this ends up in ignoring rare words which could have helped is in processing our data more efficiently.While In TfidfVectorizer we consider overall document weightage of a word. It helps us in dealing with most frequent words. Using it we can penalize them. TfidfVectorizer weights the word counts by a measure of how often they appear in the documents.

6) <b>Classifier Algoirthm</b></br>
Here we used three different Classifier Algorithm  for Classifcation. Multinomial Naive Bayes, Linear Support Vector Machine, Logistic Regression. Comparing the results of all three algorithms it concludes that for this particular problem statement Naive Bayes giving higher accuracy than the other two classification Algorithms.


