# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 16:33:04 2016

@author: Maya
"""
import nltk
from nltk.corpus import stopwords # Import the stop word list
import re,os
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
def dataPreProcessing(content):
    letters_only = re.sub("[^a-zA-Z1-9]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      content)
    lowerCase=letters_only.lower()
    tokens=nltk.word_tokenize(lowerCase)
    #print tokens
    # Remove stop words from "words"
    tokens = [w for w in tokens if not w in stopwords.words("english")]
    #print tokensStop
    porter = nltk.PorterStemmer()
    #tokens=[porter.stem(t) for t in tokens]    
    return " ".join(tokens)
        
folderNameTrue=r"C:\Users\Maya\Dropbox\mcs_ds\Project\op_spam_v1.4\positive_polarity\truthful_from_TripAdvisor\combined"
#fileName="t_hilton_1.txt"
trainTrue_X=[]
for root,dirs,fileList in os.walk(folderNameTrue):
    for fileName in fileList:
        filePath=os.path.join(root,fileName)        
        with open(filePath) as f:
            content=f.readlines()
            trainTrue_X.append(dataPreProcessing(content[0]))

noRecordsTrue=len(trainTrue_X)
print noRecordsTrue
trainTrue_y=[1 for i in range(noRecordsTrue)]
print len(trainTrue_y)

folderNameFake=r"C:\Users\Maya\Dropbox\mcs_ds\Project\op_spam_v1.4\positive_polarity\deceptive_from_MTurk\combined"
#fileName="t_hilton_1.txt"
trainFake_X=[]
for root,dirs,fileList in os.walk(folderNameFake):
    for fileName in fileList:
        filePath=os.path.join(root,fileName)        
        with open(filePath) as f:
            content=f.readlines()
            trainFake_X.append(dataPreProcessing(content[0]))
noRecordsFake=len(trainFake_X)
print noRecordsFake
trainFake_y=[0 for i in range(noRecordsFake)]
print len(trainFake_y)
#print train
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
train_X=trainTrue_X
train_X.extend(trainFake_X)
train_Y=trainTrue_y
train_Y.extend(trainFake_y)
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             ngram_range=(1, 2),
                            max_features=5000
                             )
#vectorizer = TfidfVectorizer(max_df=0.5, max_features=5000,
#                                 min_df=2, stop_words='english',
#                                 use_idf=True)
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(train_X)                             
# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()
print train_data_features.shape
# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print vocab
#print "Training the random forest..."


# Initialize a Random Forest classifier with 100 trees
#clf = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
#forest = forest.fit( train_data_features, train_Y)

#parameters = {'alpha': (0.001,0.0001,0.00001, 0.000001),
#    'penalty': ('l1','l2', 'elasticnet')
#    }
#sgd = linear_model.SGDClassifier()
#clf = GridSearchCV(sgd, parameters)

#Train a naive bayes classifier
nb=MultinomialNB()

parameters = {'alpha': (0,1),
    'fit_prior': (True, False)
    }
clf = GridSearchCV(nb, parameters,verbose=True)

clf.fit( train_data_features, train_Y)
print clf.best_estimator_
#Getting Test Data
folderNameTrueTest=r"C:\Users\Maya\Dropbox\mcs_ds\Project\op_spam_v1.4\positive_polarity\truthful_from_TripAdvisor\TestCombinedTrue"
#fileName="t_hilton_1.txt"
testTrue_X=[]
for root,dirs,fileList in os.walk(folderNameTrueTest):
    for fileName in fileList:
        filePath=os.path.join(root,fileName)        
        with open(filePath) as f:
            content=f.readlines()
            testTrue_X.append(dataPreProcessing(content[0]))

noRecordsTrue=len(testTrue_X)
print noRecordsTrue
testTrue_Y=[1 for i in range(noRecordsTrue)]

folderNameFakeTest=r"C:\Users\Maya\Dropbox\mcs_ds\Project\op_spam_v1.4\positive_polarity\deceptive_from_MTurk\TestCombinedFake"
#fileName="t_hilton_1.txt"
testFake_X=[]
for root,dirs,fileList in os.walk(folderNameFakeTest):
    for fileName in fileList:
        filePath=os.path.join(root,fileName)        
        with open(filePath) as f:
            content=f.readlines()
            testFake_X.append(dataPreProcessing(content[0]))

noRecordsFake=len(testFake_X)
print noRecordsFake
testFake_Y=[0 for i in range(noRecordsFake)]

testData_X=testTrue_X
testData_X.extend(testFake_X)
testData_Y=testTrue_Y
testData_Y.extend(testFake_Y)

test_data_features = vectorizer.transform(testData_X)
test_data_features = test_data_features.toarray()
y_pred = clf.predict(test_data_features)
print accuracy_score(testData_Y,y_pred)