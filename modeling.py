import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

def train_val_test(train, val, test, target_col):
    """
    Seperates out the target variable and creates a series with only the target variable to test accuracy.
    """
    #Seperating out the target variable
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]

    X_val = val.drop(columns = [target_col])
    y_val = val[target_col]

    X_test = test.drop(columns = [target_col])
    y_test = test[target_col]
    return X_train, y_train, X_val, y_val, X_test, y_test

def vectorize_data(X_train, X_val, X_test):
    """
    Transforms data for modeling
    """
    #Creates object
    tfidf = TfidfVectorizer()
    
    #Uses object to change data
    X_train = tfidf.fit_transform(X_train.readme_stem_no_swords)
    X_val = tfidf.transform(X_val.readme_stem_no_swords)
    X_test = tfidf.transform(X_test.readme_stem_no_swords)
    
    return X_train, X_val, X_test

    
def lr_mod(X_train, y_train, X_val, y_val, metric = 1, print_scores = False):
    """
    This function runs the Logistic Regression classifier on the training and validation test sets.
    """
    #Creating a logistic regression model
    logit = LogisticRegression(random_state=1969)

    #Fitting the model to the train dataset
    logit.fit(X_train, y_train)

    #Accuracy
    if metric == 1:
        #Make a prediction from the model
        y_pred = logit.predict(X_train)
        y_pred_val = logit.predict(X_val)        
        
        train_score = logit.score(X_train, y_train)
        val_score =  logit.score(X_val, y_val)
        method = 'Accuracy'

    
    #Precision
    elif metric == 2:
        #Make a prediction from the model
        y_pred = logit.predict(X_train)
        y_pred_val = logit.predict(X_val)

        train_score = precision_score(y_train, y_pred)
        val_score = precision_score(y_val, y_pred_val)
        method = 'Precision'

    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = logit.predict(X_train)
        y_pred_val = logit.predict(X_val)

        train_score = recall_score(y_train, y_pred)
        val_score = recall_score(y_val, y_pred_val)
        method = 'Recall'
        
    #Print the score
    if print_scores == True:
        print(f'{method} for Logistic Regression classifier on training set:   {train_score:.4f}')
        print(f'{method} for Logistic Regression classifier on validation set: {val_score:.4f}')
        #print(classification_report(y_val, y_pred_val))
    
    return train_score, val_score

def rand_forest(X_train, y_train, X_val, y_val, metric = 1, print_scores = False):
    """
    This function runs the Random Forest classifier on the training and validation test sets.
    """
    #Creating the random forest object
    rf = RandomForestClassifier(max_depth=6, 
                                class_weight = 'balanced', 
                                criterion = 'entropy',
                                n_jobs = -1,
                                min_samples_leaf = 3,
                                n_estimators = 250,
                                random_state = 77)
    
    #Fit the model to the train data
    rf.fit(X_train, y_train)

    #Accuracy
    if metric == 1:
        #Make a prediction from the model
        y_pred = rf.predict(X_train)
        y_pred_val = rf.predict(X_val)
        
        train_score = rf.score(X_train, y_train)
        val_score =  rf.score(X_val, y_val)
        method = 'Accuracy'
    
    #Precision
    elif metric == 2:
        #Make a prediction from the model
        y_pred = rf.predict(X_train)
        y_pred_val = rf.predict(X_val)

        train_score = precision_score(y_train, y_pred)
        val_score = precision_score(y_val, y_pred_val)
        method = 'Precision'
        
    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = rf.predict(X_train)
        y_pred_val = rf.predict(X_val)

        train_score = recall_score(y_train, y_pred)
        val_score = recall_score(y_val, y_pred_val)
        method = 'Recall'
        
    #Print the score
    if print_scores == True:
        print(f'{method} for Random Forest classifier on training set:   {train_score:.4f}')
        print(f'{method} for Random Forest classifier on validation set: {val_score:.4f}')
        #print(classification_report(y_val, y_pred_val))

    return train_score, val_score

def dec_tree(X_train, y_train, X_val, y_val, metric = 1, print_scores = False):
    """
    This function runs the Decission Tree classifier on the training and validation test sets.
    """
    #Create the model
    clf = DecisionTreeClassifier(max_depth=5, random_state=1969)
    
    #Train the model
    clf = clf.fit(X_train, y_train)
    
    #Accuracy
    if metric == 1:
        #Make a prediction from the model
        y_pred = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)
        
        train_score = clf.score(X_train, y_train)
        val_score =  clf.score(X_val, y_val)
        method = 'Accuracy'
    #Precision
    elif metric == 2:
        #Make a prediction from the model
        y_pred = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)

        train_score = precision_score(y_train, y_pred)
        val_score = precision_score(y_val, y_pred_val)
        method = 'Precision'
        
    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)

        train_score = recall_score(y_train, y_pred)
        val_score = recall_score(y_val, y_pred_val)
        method = 'Recall'
        
    #Print the score
    if print_scores == True:
        print(f'{method} for Decision Tree classifier on training set:   {train_score:.4f}')
        print(f'{method} for Decision Tree classifier on validation set: {val_score:.4f}')
        #print(classification_report(y_val, y_pred_val))
    
    return train_score, val_score
