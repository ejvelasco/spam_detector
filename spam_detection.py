import email
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
# set up data
email_data_path = 'spam/data/'
email_labels_path = 'spam/data.labels'
email_text = []
email_labels = []
# parse email labels
with open (email_labels_path, encoding='latin-1') as labels:
	for label in labels:
		email_labeled = label.strip().split(' ')
		email_labels.append(int(email_labeled[0]))
# parse email text
for file_name in sorted(os.listdir(email_data_path)):
	with open(email_data_path + file_name, encoding='latin-1') as eml_data:
		parsed_email = email.message_from_file(eml_data)
		text = ''
		for section in parsed_email.walk():
			section_type = section.get_content_type() 
			if section_type == 'text/html' or section_type == 'text/plain':
				text += section.get_payload()
		email_text.append(text)
# compute model evaluation scores
def comp_scores(clf, data, labels):
	data_processed = pipeline.fit_transform(data)
	accuracies = cross_val_score(clf, data_processed, labels, cv=cv, scoring='accuracy')
	accuracy_avg = np.sum(accuracies)/cv
	predictions = cross_val_predict(clf, data_processed, labels, cv=cv)
	precision = precision_score(labels, predictions)
	recall =  recall_score(labels, predictions)
	scores = [
		accuracy_avg, 
		precision,
		recall
	]
	return list(map(lambda num: float(str(num*100)[:5]), scores))
# split data
email_text_train, email_text_test = train_test_split(email_text, test_size=0.2, random_state=42)
email_labels_train, email_labels_test = train_test_split(email_labels, test_size=0.2, random_state=42)
# process data, train model and test
cv = 3
sgd_clf = SGDClassifier(random_state=42)
pipeline = Pipeline([
	('vectorizer', CountVectorizer()),
])
train_scores = comp_scores(sgd_clf, email_text_train, email_labels_train)
test_scores = comp_scores(sgd_clf, email_text_test, email_labels_test)
print(train_scores)
print(test_scores)