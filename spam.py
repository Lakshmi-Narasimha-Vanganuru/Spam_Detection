# PROJECT NAME - EMAIL SPAM DETECTION 
# it is used as supervised learning (only works on the dataset)

import numpy as np # need to creating numpy arrays so use numpy as called as np(short-form os numpy) for further uses
import pandas as pd # used  to create DataFrames(in simply,used to structure the data) using pandas
from sklearn.model_selection import train_test_split # used to train the model selection using the train_test_split function
from sklearn.feature_extraction.text import TfidfVectorizer # used to extract(bcz of this TfidfVectorizer, we convert text data i.e SPAM or HAM to Binary i.e 0's and 1's) text from the feature extraction function
from sklearn.linear_model import LogisticRegression # in simply, use this model to classify the model is spam or ham
from sklearn.metrics import accuracy_score #  this is used to evaluate the modle (i.e how it performs based on test data)

#Data collection and Pre-Processing(load the data using pandas)
raw_mail_data = pd.read_csv("C:/Users/naras/OneDrive/Desktop/uio/mail.csv") # read_csv is used to load the data from the mail

# checking for missing-data (WHY? , bcz missing-data leads to unexpected outputes so ,replace the null values with a null string)
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'') # Here, 'where' is used as a condition for missing-values and set as empty or null-string

# set label-Data spam mail as 0 & ham mail as 1( WHY? ,bcz for classifying/Predicting the output more easily)

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 1
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 0

# separating the data as texts and label(i.e X & Y)
# (WHY? bcz of easy way to distinguish between messages and category)
X = mail_data['Message'] # set message as -> 'X'

Y = mail_data['Category'] # set category as -> 'Y'

# splitting data into train and test ( WHY? , for machine training and model building)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)# Test to the test_data is done in 20% , random_state used split the data exactly (3)
print("training data full ,containing Rows and Cloumn is:",X.shape) #no. of rows is 5572 ,and column is 1
print("After training 80% of data is:",X_train.shape)  
print("After testing 20% of data is:",X_test.shape)

# transform the text data to feature vectors that can be used as input to the Logistic regression (vector is binary format)
# WHY? ,bcz need to convert the text data into the binary data (i.e o's and 1's) using -> 'TfidfVectorizer function'

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True) # min_df is give min score is given to a particular word(in mail dataset),stop_words used to ignore the common words like "are,and,is,at ..."

# now convert the feature_extraction into data
X_train_features = feature_extraction.fit_transform(X_train) # converted all data into binary stored into -> "x_,i.e fitisto fit the data and extracted into binary
X_test_features = feature_extraction.transform(X_test)#based on fit data , do test on x_train

# convert Y_train and Y_test values as integers (in dataset objectdatatype is strings in some values considered as  strings so need to convert into integers)
#  ( WHY? ,bcz in output prediction the model is used to classify based on integers values only)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# MODEL TRAINING
model = LogisticRegression() # WHY? used logistic regression,bcz it is good at classification in binary classification

# training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train) # X_train_features, Y_train both are in binary so easy for model to understand

# evaluate the model (i.e how it performs based on test data)
# prediction on training data

prediction_on_training_data = model.predict(X_train_features) # predict used to give output as 0 or 1 (i.e spam or not)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data) # getting the accuray ,how much is good at.
print('Accuracy on training data : ', accuracy_on_training_data) # Accuracy on training data :  0.9676912721561588

# prediction on test data (same as above)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data : ', accuracy_on_test_data) # Accuracy on test data :  0.9668161434977578

# Making prediction on new mail (input mail)

input_mail = ["free money"]
# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction system

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('spam mail')

else:
  print('Ham Mail')