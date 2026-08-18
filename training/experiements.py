import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

load_dotenv()
seed = int(os.getenv("SEED"))

# loading the cleaned datasets 
train_df = pd.read_csv("train_clean.csv")
validation_df = pd.read_csv("validation_clean.csv")
test_df = pd.read_csv("test_clean.csv")

print("Clean datasets Loaded successfully ... ")

# define feature and target varaibles
X_train = train_df['text']
y_train = train_df['label']

# remember to keep this test aside and never show it to the model untill we select the model and the hyperparameters for the model. This is to avoid data leakage and overfitting.
X_test = test_df['text']
y_test = test_df['label']

X_val = validation_df['text']
y_val = validation_df['label']

# dummy classifier 
dclf = DummyClassifier(strategy='most_frequent')
dclf.fit(X_train, y_train)

y_pred_dclf = dclf.predict(X_val)

# claculate the accuracy for the dummy classifier 
print(f"The accuracy of the dummy classifier, strategy : most_frequent is {accuracy_score(y_true = y_val, y_pred = y_pred_dclf)}")

# lgositic regression - untuned

lr_pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(random_state=seed, max_iter=1000))

# fit the pipeline on the traingin set 
lr_pipeline.fit(X_train, y_train)

y_pred_pipe = lr_pipeline.predict(X_val)

#calculate the accuracy score for the untuned logistic regression model 
print(f"The accuracy of the untuned logistic regression, strategy : most_frequent is {accuracy_score(y_true = y_val, y_pred = y_pred_pipe)}")

print("Classifiction report of the Dummy classifier ... ")

print(classification_report(y_true=y_val, y_pred=y_pred_dclf))

print("Classifiction report of the untuned logistic regression pipeline ... ")

print(classification_report(y_true=y_val, y_pred=y_pred_pipe))

# Applying stopwords (to see if stripping stop words -common english words- helps the model)

print("Applying stop words Removal")

exp_a_pipe = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression(random_state=seed, max_iter=1000))
exp_a_pipe.fit(X_train, y_train)
y_pred_exp_a = exp_a_pipe.predict(X_val)
print("classification report for untuned logisitic regression - Exp.A stripping stop words from the text")
print(classification_report(y_true=y_val, y_pred=y_pred_exp_a))


# Applying stopwords (to see if stripping stop words -common english words- helps the model)

print("Applying N-grams - giving the model the ability to read combined words")

exp_b_pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(random_state=seed, max_iter=1000))
exp_b_pipe.fit(X_train, y_train)
y_pred_exp_b = exp_b_pipe.predict(X_val)
print("classification report for untuned logisitic regression - Exp.B applying N-gram (1, 2)")
print(classification_report(y_true=y_val, y_pred=y_pred_exp_b))

# we will go further with exp_b_pipe (selected and locked model)

max_features_options = [3000,5000,10000,None]

for max_feature in max_features_options:
    sweep_pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), stop_words=None, max_features=max_feature),
                               LogisticRegression(random_state=seed, max_iter=1000))
    sweep_pipe.fit(X_train, y_train)
    y_pred_sweep = sweep_pipe.predict(X_val)
    print(f"Classification matrix for our sweep pipleline when max_feautres is {max_feature} is :")
    print(classification_report(y_true = y_val, y_pred = y_pred_sweep))
    


