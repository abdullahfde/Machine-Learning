import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,VotingClassifier
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib



from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle

mnist = fetch_mldata('MNIST original')
X = mnist['data']
y = mnist['target']

X_train, X_validation_test, y_train, y_validation_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Training set
X_train, y_train = shuffle(X_train, y_train, random_state=0)

X_validation_test, y_validation_test = shuffle(X_validation_test, y_validation_test, random_state=0)

# Validation set
X_validation, y_validation, = X_validation_test[:5000], y_validation_test[:5000]

# Test set
X_test, y_test = X_validation_test[5000:], y_validation_test[5000:]



rnd_clf	=	RandomForestClassifier(random_state=0)
rnd_clf.fit(X_train,y_train)
joblib.dump(rnd_clf, 'RFClassifier.pkl' )


extra_clf=ExtraTreesClassifier(random_state=0)
extra_clf.fit(X_train,y_train)
joblib.dump(extra_clf, 'ETClassifier.pkl' )


rnd_clf_1=RandomForestClassifier(random_state=0)
extra_clf_1=ExtraTreesClassifier(random_state=0)
voting_clf=VotingClassifier(estimators=[('rf',	rnd_clf_1),('et',extra_clf_1)],voting='soft')
voting_clf.fit(X_train,y_train)
joblib.dump(voting_clf, 'SoftEnsembleClassifier.pkl')


accuracy=[]
y_pred_rnd = rnd_clf.predict(X_test)
accuracy_rnd=accuracy_score(y_test, y_pred_rnd)

y_pred_extra = extra_clf.predict(X_test)
accuracy_extra=accuracy_score(y_test, y_pred_extra)

y_pred_voting = voting_clf.predict(X_test)
accuracy_voting=accuracy_score(y_test, y_pred_voting)

accuracy.append(accuracy_rnd)
accuracy.append(accuracy_extra)
accuracy.append(accuracy_voting)
joblib.dump(accuracy, 'part_d.pkl')



y_pred_rnd_clf=rnd_clf.predict_proba(X_validation)
y_pred_extra_clf=extra_clf.predict_proba(X_validation)
merged= []
new_training=np.array([])
for i in range (5000):
    merged.append(y_pred_rnd_clf[i].tolist()+y_pred_extra_clf[i].tolist())
new_training=np.array(merged)
joblib.dump(new_training, 'part_e.pkl')





rnd_clf_new=RandomForestClassifier(random_state=0)
rnd_new= rnd_clf_new.fit(new_training,y_validation)
joblib.dump(rnd_new, 'Blender.pkl')



y_pred_rnd_clf_x_test=rnd_clf.predict_proba(X_test)
y_pred_extra_clf_x_test=extra_clf.predict_proba(X_test)
merged_= []
new_x_test=np.array([])
for i in range (5000):
    merged_.append(y_pred_rnd_clf_x_test[i].tolist()+y_pred_extra_clf_x_test[i].tolist())
new_x_test=np.array(merged_)
y_pred_last=rnd_new.predict(new_x_test)
final_accuracy=accuracy_score(y_test,y_pred_last)

joblib.dump(final_accuracy, 'part_g.pkl')






