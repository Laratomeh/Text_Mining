#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
import pandas as pd
import seaborn as sns
import contractions
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.models import *
from keras.layers import *


# In[2]:


# Importing the datasets
train_data = pd.read_csv(r'Lyrics-Genre-Train.csv')
test_data = pd.read_csv(r'Lyrics-Genre-Test-GroundTruth.csv')


# In[3]:


# Removing irrelevant columns from training and testing data
train_data = train_data.drop(['Song', 'Song year', 'Artist', 'Track_id'], axis=1)
test_data = test_data.drop(['Song', 'Song year', 'Artist', 'Track_id'], axis=1)


# In[4]:


# Plotting songs count by genre for both datasets
plt.figure(figsize=(8,7))
plt.subplot(211)
sns.countplot(x='Genre', data=train_data, color='skyblue')
plt.title('Song Count by Genre in Training Data')
plt.ylabel('Number of Songs', fontsize=12)
plt.xlabel('Genre', fontsize=12)
plt.show()
plt.figure(figsize=(8,7))
plt.subplot(212)
sns.countplot(x='Genre', data=test_data, color='skyblue')
plt.title('Song Count by Genre in Testing Data')
plt.ylabel('Number of Songs', fontsize=12)
plt.xlabel('Genre', fontsize=12)
plt.show()


# In[5]:


train_data.drop(train_data.index[train_data['Genre'] == 'Folk'], inplace=True)
train_data.drop(train_data.index[train_data['Genre'] == 'R&B'], inplace=True)
train_data.drop(train_data.index[train_data['Genre'] == 'Indie'], inplace=True)

test_data.drop(test_data.index[test_data['Genre'] == 'Folk'], inplace=True)
test_data.drop(test_data.index[test_data['Genre'] == 'R&B'], inplace=True)
test_data.drop(test_data.index[test_data['Genre'] == 'Indie'], inplace=True)


# In[6]:


def recall_score(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
def precision_score(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[7]:


def clean(text):
    # convert text to lowercase
    text = text.lower()  
    # git red of non word carachters
    text = re.sub(r'\W', ' ', text)
    # remove digits
    text = re.sub(r'\d', ' ', text)
    # remove single carachters
    text = re.sub(r'\s+[a-z]\s+', ' ', text, flags=re.I)
    # remove single carachters at the start of the sentence 
    text = re.sub(r'^[a-z]\s+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text) 
    text = re.sub(r'^\s', '', text) # space at beggining
    text = re.sub(r'\s$', '', text) # space at ending
    # Removing contractions "abbreviations"
    text = contractions.fix(text)
    # get rid of stopwords
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
    # # Stemming words
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    # Lemmatizing words
#     lemmatizer = WordNetLemmatizer()
#     text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text


# In[8]:


# Apply clean function to the text
train_data["Lyrics"] = train_data["Lyrics"].apply(clean)
test_data["Lyrics"] = test_data["Lyrics"].apply(clean)


# In[9]:


df_train, df_val = train_test_split(train_data, test_size=0.2, random_state=42, shuffle=True)
df_train.reset_index()  


# In[10]:


X_train = df_train['Lyrics'].values
X_val = df_val['Lyrics'].values
X_test = test_data['Lyrics'].values


# In[11]:


def genre_encode(Genre):
    """
    return one hot encoding for Y value
    """
    if Genre == 'Pop':
        return 0
    elif Genre == 'Country':
        return 1
    elif Genre == 'Rock':
        return 2
    elif Genre == 'Hip-Hop':
        return 3
    elif Genre == 'Metal':
        return 4
    elif Genre == 'Jazz':
        return 5
    else:
        return 6


# In[12]:


genres = df_train['Genre'].tolist()
y_train = [genre_encode(genre) for genre in genres]
y_train = np.array(y_train)

genres = df_val['Genre'].tolist()
y_val = [genre_encode(genre) for genre in genres]
y_val = np.array(y_val)

genres = test_data['Genre'].tolist()
y_test = [genre_encode(genre) for genre in genres]
y_test = np.array(y_test)


# In[13]:


vectorizer = CountVectorizer(max_features=20000)
X_train = vectorizer.fit_transform(X_train)
X_val = vectorizer.transform(X_val)
X_test = vectorizer.transform(X_test)


# In[14]:


print('X_train shape :', X_train.shape)
print('X_val shape :', X_val.shape)
print('X_test shape :', X_test.shape)


# # ML Models

# ##  MultinomialNB Model

# In[15]:


mn_model = MultinomialNB(alpha=.0433)
mn_model.fit(X_train, y_train)


# In[16]:


accuracy = mn_model.score(X_train, y_train)
print(f'Accuracy on Training set: {accuracy:.5f}')


# In[17]:


mn_pred = mn_model.predict(X_val)
accuracy = mn_model.score(X_val, y_val)
print(f'Accuracy on Validation set: {accuracy:.5f}')


# In[18]:


cm = confusion_matrix(y_val, mn_pred, labels=mn_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=mn_model.classes_)
disp.plot()
plt.show()


# In[19]:


mn_pred = mn_model.predict(X_test)
accuracy = mn_model.score(X_test, y_test)
print(f'Accuracy on Test set: {accuracy:.5f}')


# In[20]:


cm = confusion_matrix(y_test, mn_pred, labels=mn_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=mn_model.classes_)
disp.plot()
plt.show()


# ## Random Forest Classifier Model 

# In[21]:


random_forest_model = RandomForestClassifier(n_estimators=300, criterion ='gini', max_depth = 20) 
random_forest_model = random_forest_model.fit(X_train,y_train)


# In[22]:


accuracy = random_forest_model.score(X_train, y_train)
print(f'Accuracy on Training set: {accuracy:.5f}')


# In[23]:


rf_pred = random_forest_model.predict(X_val)
accuracy = random_forest_model.score(X_val, y_val)
print(f'Accuracy on Validation set: {accuracy:.5f}')


# In[24]:


cm = confusion_matrix(y_val, rf_pred, labels=random_forest_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=random_forest_model.classes_)
disp.plot()
plt.show()


# In[25]:


rf_pred = random_forest_model.predict(X_test)
accuracy = random_forest_model.score(X_test, y_test)
print(f'Accuracy on Test set: {accuracy:.5f}')


# In[26]:


cm = confusion_matrix(y_test, rf_pred, labels=random_forest_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=random_forest_model.classes_)
disp.plot()
plt.show()


# ## SVC Model

# In[27]:


svc_model = SVC() 
svc_model = svc_model.fit(X_train,y_train)


# In[28]:


accuracy = svc_model.score(X_train, y_train)
print(f'Accuracy on Training set: {accuracy:.5f}')


# In[29]:


svc_pred = svc_model.predict(X_val)
accuracy = svc_model.score(X_val, y_val)
print(f'Accuracy on Validation set: {accuracy:.5f}')


# In[30]:


cm = confusion_matrix(y_val, svc_pred, labels=svc_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=svc_model.classes_)
disp.plot()
plt.show()


# In[31]:


svc_pred = svc_model.predict(X_test)
accuracy = svc_model.score(X_test, y_test)
print(f'Accuracy on Test set: {accuracy:.5f}')


# In[32]:


cm = confusion_matrix(y_test, svc_pred, labels=svc_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=svc_model.classes_)
disp.plot()
plt.show()


# ## AdaBoost Classifier

# In[33]:


AdaBoostClassifier_model = AdaBoostClassifier() 
AdaBoostClassifier_model = AdaBoostClassifier_model.fit(X_train,y_train)


# In[34]:


accuracy = AdaBoostClassifier_model.score(X_train, y_train)
print(f'Accuracy on Training set: {accuracy:.5f}')


# In[35]:


AdaBoostClassifier_pred = AdaBoostClassifier_model.predict(X_val)
accuracy = AdaBoostClassifier_model.score(X_val, y_val)
print(f'Accuracy on Validation set: {accuracy:.5f}')


# In[36]:


cm = confusion_matrix(y_val, AdaBoostClassifier_pred, labels=AdaBoostClassifier_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=AdaBoostClassifier_model.classes_)
disp.plot()
plt.show()


# In[37]:


AdaBoostClassifier_pred = AdaBoostClassifier_model.predict(X_test)
accuracy = AdaBoostClassifier_model.score(X_test, y_test)
print(f'Accuracy on Test set: {accuracy:.5f}')


# In[38]:


cm = confusion_matrix(y_test, AdaBoostClassifier_pred, labels=AdaBoostClassifier_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=AdaBoostClassifier_model.classes_)
disp.plot()
plt.show()


# # Deep Learning Models

# In[39]:


y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

X_train.sort_indices()
X_val.sort_indices()
X_test.sort_indices()


# In[45]:


dropout = 0.3
batch_size = 64
epochs = 15

# Model 1
# Build the model
model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(dropout))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(dropout))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(dropout))
model.add(Dense(16, activation = 'relu'))
model.add(Dropout(dropout))
model.add(Dense(7, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_score])
model.summary()

# # # Model 2
# # Build the model
# model = keras.Sequential()
# model.add(keras.layers.Dense(64, input_dim=X_train.shape[1], activation='relu'))
# model.add(keras.layers.Dense(64, activation='relu'))
# model.add(keras.layers.Dense(7, activation='softmax'))
# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_score])
# model.summary()


# In[46]:


# fitting the model
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))


# In[47]:


print('Training Accuracy')
a=model.evaluate(X_train,y_train)
print('Loss: {} \nAccuracy: {} \nF-1 Score: {}'.format(a[0],a[1],a[2]))


# In[48]:


print('Validation Accuracy')
a=model.evaluate(X_val,y_val)
print('Loss: {} \nAccuracy: {} \nF-1 Score: {}'.format(a[0],a[1],a[2]))


# In[49]:


print('Test Accuracy')
a=model.evaluate(X_test,y_test)
print('Loss: {} \nAccuracy: {} \nF-1 Score: {}'.format(a[0],a[1],a[2]))

