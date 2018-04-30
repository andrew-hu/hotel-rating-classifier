import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from numpy import genfromtxt
import time
import json

input_file = "preprocessed.csv"

# Load Hotel_reviews data
df = pd.read_csv(input_file, header = 0)
data_pandas = pd.DataFrame(df)

train = df[df['is_train']==True]
test = df[df['is_train']==False]

trainTargets = np.array(train['processed_score']).astype(int)
testTargets = np.array(test['processed_score']).astype(int)

# columns you want to model
features = df.columns[20:159]

# call Gaussian Naive Bayesian class with default parameters
gnb = GaussianNB()

# train model
#y_gnb = gnb.fit(train[features], trainTargets).predict(train[features])
#y_gnb = gnb.fit(train[features], trainTargets).predict(test[features])

#print(accuracy_score(testTargets, y_gnb))
model = gnb.fit(train[features], trainTargets)
# List of names
mylist = list(df)
del mylist[0:20]

while True:
	predict_list = []
	# Read review string from JSON
	with open('review.txt') as json_data:
		d = json.load(json_data)
		review = d["review"]

	# Split review string into tokens
	review_words = review.split()

	# Check if word from d array are present in 
	for word in mylist:
		if word in review_words:
			predict_list.append(1)
		else:
			predict_list.append(0)	
	predicted = model.predict([predict_list])	
	print(predicted)
	result = { "stars": predicted.item(0) }

	with open('predicted.txt',"w") as outfile:    
		json.dump(result, outfile, indent=4)
		
	time.sleep(2)