import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_true = pd.read_csv('true.csv')
df_fake = pd.read_csv('fake.csv')

print(df_fake.head())
print(df_true.head())

print(df_fake.isnull().sum())
print(df_true.isnull().sum())

print(df_fake.describe().transpose())
print(df_true.describe().transpose())

df_fake['label'] = 0  
df_true['label'] = 1  

df = pd.concat([df_fake, df_true], ignore_index=True)

print("Duplicate titles:", df.duplicated(subset='title').sum())
print("Duplicate texts:", df.duplicated(subset='text').sum())

#Drop Duplicate 
df = df.drop_duplicates(subset='text')
df['length'] = df['text'].apply(len)


#Data Visualization
df.hist(column='length', by='label', bins= 40, figsize=(12,8))
plt.show()

df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
df.hist(column='word_count', by='label', bins=40, figsize=(12,8))
plt.suptitle('Word Count Distribution by News Type')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()

sns.countplot(x='label', data=df)
plt.xticks([0, 1], ['Fake', 'True'])
plt.title('Class Distribution')
plt.show()

#Data Preprocessing
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))
    
    text = re.sub(r'\d+', '', text)
    
    tokens = text.split()
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)


df['clean_text'] = df['text'].apply(preprocess_text)

print(df[['text', 'clean_text']].head())


#Text Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words = 'english') 

X_text = df['clean_text']
y_label = df['label']

#Text Preprocessing
from sklearn.model_selection import train_test_split

x_train_text,x_test_text,y_train,y_test = train_test_split(X_text, y_label , test_size=0.2, random_state=101)

#Data Scaling
x_train = vectorizer.fit_transform(x_train_text)
x_test = vectorizer.transform(x_test_text)

#Model Creation 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)



model = Sequential()
model.add(Dense(76, activation='relu', input_dim=x_train.shape[1]))
model.add(Dropout(0.3))

model.add(Dense(57, activation = 'relu'))
model.add(Dropout(0.3))

model.add(Dense(38, activation = 'relu'))
model.add(Dropout(0.3))

model.add(Dense(19, activation = 'relu'))
model.add(Dropout(0.3))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()

#Model Scaling
model.fit(x = x_train, y = y_train, epochs = 400, validation_data = (x_test, y_test), callbacks = [early_stop] )

#Loss Visualization 
model_loss = pd.DataFrame(model.history.history)
model_loss.plot(figsize=(12,8))
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.grid(True)
plt.show()

#Evaluate the model
prediction = (model.predict(x_test)> 0.5).astype (int)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, prediction))
print('\n')
print(classification_report(y_test,prediction))
print("Accuracy:", accuracy_score(y_test, prediction))

#Prediction 
X_test_text = x_test_text.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

for i in range(5):
    print(f"Message: {X_test_text[i]}")
    print(f"Actual: {y_test[i]}, Predicted: {int(prediction[i])}")
    print('-'*50)

