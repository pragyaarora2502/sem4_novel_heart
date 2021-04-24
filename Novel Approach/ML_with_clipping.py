import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


data = pd.read_csv("Features.csv")

print("\nBasic Description of Data\n")
print("\nData Shape")
print(data.shape)

print("\nTarget Variable types")
print(data['type'].value_counts())

### create a column to identify if the audio clip is normal or not
data.loc[data['type'] != 'normal', 'normal'] = 0
data.loc[data['type'] == 'normal', 'normal'] = 1

print("\nValue counts of records having normal heartbeat")
print(data['normal'].value_counts())

### encoding the data into integers
data.loc[data['type'] == 'normal', 'type'] = 0
data.loc[data['type'] == 'murmur', 'type'] = 1
data.loc[data['type'] == 'artifact', 'type'] = 2
data.loc[data['type'] == 'extrastole', 'type'] = 3
data.loc[data['type'] == 'extrahls', 'type'] = 4



### Data preprocessing for ML
X = data.iloc[:, 0:34]
y = data['normal']

print("\n\nTrain test stratified split with 25% ratio")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9, stratify=y)
print("\nTrain size: ", len(X_train))
print("Test size: ", len(X_test))
print("\nTarget Variable counts in Train data: ", pd.DataFrame(y_train).value_counts())

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)



### Model building to perform Binary classification on 'Normal" column
## Random Forest Classifier
print("\n\nRandom Forest Classifier: ")
forest = RandomForestClassifier(random_state=9, oob_score=True, class_weight='balanced')
forest.fit(X_train, y_train)
print("\nOOB score: ", forest.oob_score_)

y_pred = forest.predict(X_train)
print("\nTrain Accuracy Score: ", accuracy_score(y_train, y_pred))
print("Train data Confusion Matrix: \n", confusion_matrix(y_train,y_pred))

y_pred = forest.predict(X_test)
print("\nTest Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Test data Confusion Matrix: \n", confusion_matrix(y_test,y_pred))

y_pred = forest.predict(X)
print("\nComplete data Accuracy Score: ", accuracy_score(y, y_pred))
print("Complete data Confusion Matrix: \n", confusion_matrix(y,y_pred))

y_pred = forest.predict(X_test)
print("\nClassification report on test data: \n", classification_report(y_test, y_pred))

pickle.dump(forest, open("clips_layer_1_RF.sav", 'wb'))
print("\nModel saved as clips_layer_1_RF.sav")


## Gradient Boosting Classifier
print("\n\nGradient Boosting Classifier: ")
gradient = GradientBoostingClassifier(random_state=9)
gradient.fit(X_train, y_train)

y_pred = gradient.predict(X_train)
print("\nTrain Accuracy Score: ", accuracy_score(y_train, y_pred))
print("Train data Confusion Matrix: \n", confusion_matrix(y_train,y_pred))

y_pred = gradient.predict(X_test)
print("\nTest Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Test data Confusion Matrix: \n", confusion_matrix(y_test,y_pred))

y_pred = gradient.predict(X)
print("\nComplete data Accuracy Score: ", accuracy_score(y, y_pred))
print("Complete data Confusion Matrix: \n", confusion_matrix(y,y_pred))

y_pred = gradient.predict(X_test)
print("\nClassification report on test data: \n", classification_report(y_test, y_pred))

pickle.dump(gradient, open("clips_layer_1_GB.sav", 'wb'))
print("\nModel saved as clips_layer_1_GB.sav")



### Model building to perform multi class classification on 'Type' column

new_data = data[data['normal']==0]
X = new_data.iloc[:, 0:34]
y = new_data['type'].astype('int')

print("\n\n\nTrain test stratified split with 25% ratio")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9, stratify=y)
print("\nTrain size: ", len(X_train))
print("Test size: ", len(X_test))
print("\nTarget Variable counts in Train data: ", pd.DataFrame(y_train).value_counts())

## Random Forest Classifier
print("\n\nRandom Forest Classifier: ")
forest = RandomForestClassifier(random_state=9, oob_score=True, class_weight='balanced')
forest.fit(X_train, y_train)
print("\nOOB score: ", forest.oob_score_)

y_pred = forest.predict(X_train)
print("\nTrain Accuracy Score: ", accuracy_score(y_train, y_pred))
print("Train data Confusion Matrix: \n", confusion_matrix(y_train,y_pred))

y_pred = forest.predict(X_test)
print("\nTest Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Test data Confusion Matrix: \n", confusion_matrix(y_test,y_pred))

y_pred = forest.predict(X)
print("\nComplete data Accuracy Score: ", accuracy_score(y, y_pred))
print("Complete data Confusion Matrix: \n", confusion_matrix(y,y_pred))

y_pred = forest.predict(X_test)
print("\nClassification report on test data: \n", classification_report(y_test, y_pred))

pickle.dump(forest, open("clips_layer_2_RF.sav", 'wb'))
print("\nModel saved as clips_layer_2_RF.sav")


## Gradient Boosting Classifier
print("\n\nGradient Boosting Classifier: ")
gradient = GradientBoostingClassifier(random_state=9)
gradient.fit(X_train, y_train)

y_pred = gradient.predict(X_train)
print("\nTrain Accuracy Score: ", accuracy_score(y_train, y_pred))
print("Train data Confusion Matrix: \n", confusion_matrix(y_train,y_pred))

y_pred = gradient.predict(X_test)
print("\nTest Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Test data Confusion Matrix: \n", confusion_matrix(y_test,y_pred))

y_pred = gradient.predict(X)
print("\nComplete data Accuracy Score: ", accuracy_score(y, y_pred))
print("Complete data Confusion Matrix: \n", confusion_matrix(y,y_pred))

y_pred = gradient.predict(X_test)
print("\nClassification report on test data: \n", classification_report(y_test, y_pred))

pickle.dump(gradient, open("clips_layer_2_GB.sav", 'wb'))
print("\nModel saved as clips_layer_2_GB.sav")


'''
Way to use model in another file: 

import pickle
model = pickle.load(open("model name", "rb"))
'''