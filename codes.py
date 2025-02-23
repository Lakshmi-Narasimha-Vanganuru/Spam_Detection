#Spam Detection Project using ml
import numpy as np  # NumPy is used for numerical operations
import pandas as pd  # Pandas is used for handling datasets (CSV files)

# Importing necessary modules for machine learning
from sklearn.model_selection import train_test_split  # Used to split dataset into training and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text data into numerical features (TF-IDF)
from sklearn.linear_model import LogisticRegression  # Logistic Regression model for spam classification
from sklearn.metrics import accuracy_score  # To evaluate model performance
import pickle  # Used to save the trained model and vectorizer

<<<<<<< HEAD:codes.py
# **STEP 1: Load Dataset**
raw_mail_data = pd.read_csv("mail.csv")  
=======
#Load Dataset
raw_mail_data = pd.read_csv("C:/Users/naras/OneDrive/Desktop/uio/mail.csv")  
>>>>>>> 40997af3dd1a2a046580af7ba8eac1520256b6ce:codes.py.py
# Checking for missing values and replacing them with empty strings
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')  

# STEP 2: Encode Labels
# Spam = 1 (positive class), Ham = 0 (negative class)
# WHY? Because ML models understand numerical values, not text labels
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 1  
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 0  

# STEP 3: Splitting the Dataset
X = mail_data['Message']  # Features (email text)
Y = mail_data['Category'].astype(int)  # Labels (spam or ham)
# Split data into 80% training and 20% testing (random_state=42 ensures consistent results)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# STEP 4: Convert Text Data to Numeric Data Using TF-IDF
# WHY? Machine learning models cannot process raw text directly, so we convert them into numerical vectors
vectorizer = TfidfVectorizer(
    min_df=1,  # Ignore terms that appear in only one document (reduces noise)
    stop_words='english',  # Removes common words like "the", "is", "and" to improve model performance
    lowercase=True,  # Converts text to lowercase to maintain uniformity
    ngram_range=(1,2),  # Includes both individual words and word pairs (bigrams) to capture more context
    max_features=5000  # Limits the number of features to the top 5000 most important words
)  

# Transform training and test text data into numerical vectors
X_train_features = vectorizer.fit_transform(X_train)  
X_test_features = vectorizer.transform(X_test)  

# STEP 5: Train the Model (Logistic Regression)
# WHY Logistic Regression? It is simple, efficient, and works well for binary classification problems like spam detection
model = LogisticRegression(
    max_iter=500,  # Increases iterations to ensure the model converges properly
    class_weight='balanced',  # Adjusts weight for imbalanced data (spam is usually less than ham)
    solver='lbfgs'  # Optimized solver for handling large datasets
)  

# Train the model using the training data
model.fit(X_train_features, Y_train)  

# STEP 6: Evaluate Model Performance
# WHY? To check how well the model has learned from the training data
train_accuracy = accuracy_score(Y_train, model.predict(X_train_features)) * 100  
test_accuracy = accuracy_score(Y_test, model.predict(X_test_features)) * 100  

# Print the accuracy scores (with 2 decimal precision)
print(f'Accuracy on training data: {train_accuracy:.2f}%')  
print(f'Accuracy on test data: {test_accuracy:.2f}%')  

# STEP 7: Save the Model and Vectorizer
with open("spam_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# STEP 8: Define Function to Predict New Messages
def predict_spam(message, vectorizer, model):
    """
    This function takes a new email message as input, 
    processes it using the trained model, and predicts whether it is spam or ham.
    It also provides the probability of the message being spam.
    """
    # Convert input message to numerical features using the trained vectorizer
    input_features = vectorizer.transform([message])
    
    # Predict spam or ham (1 = Spam, 0 = Ham)
    prediction = model.predict(input_features)[0]
    
    # Get probability score for spam classification
    prediction_proba = model.predict_proba(input_features)[0][1] * 100  # Convert to percentage
    
    # Print the prediction and probability
    print(f"\nMessage: {message}")
    print(f"Prediction: {'Spam' if prediction == 1 else 'Ham'}")
    print(f"Prediction Probability: {prediction_proba:.2f}%")  # Shows confidence in the prediction

# **STEP 9: Test with a New Email Message**
new_mail = "free money"
predict_spam(new_mail, vectorizer, model)
