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
Before doing analysis on data we need to transform the dataset and clean it so that the computer is able to understand that data in the desired format.Data preprocessing is a fundamental step while building a machine learning model. If the data is fairly pre-processed the results would be reliable. In NLP, the first step before building the machine learning model, is to pre-process the data.

In this particular Solution we have used various text cleaning methods like </br>
1) stop words
2) lemmatization
3) stemming algorithm


It is trade between speed and accuracy to Choose between stemming and lemmatization because in Stemming algorithm,it is used to extract the base form of the words by removing affixes from them. It is just like cutting down the branches of a tree to its stems. The stemmed word may or may not be in the context of dictionary.While in Lemmatization, words are stemmed to its root word which will be form the dictionary. Lemmatization takes more time in cleaning but accuracy is high and Stemming is purely opoosite to this, it gives priority to speed over accuracy.

Here we have used Lemmatization for high accuracy.

4) Dividing the dataset into X_train, y_train, X_test, y_test.

5) <b>Word Vectorization</b>

