import numpy as np
from PIL import Image
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import time
from sklearn.linear_model import LogisticRegression
import os

path_train = "path"
path_test = "path"
images_path_train = []
images_path_test = []

images_train = []
images_test = []

for root, dirs, files in os.walk(path_train):
    for dirname in sorted(files):
        splitname = dirname.split('.')
        if len(splitname) > 1:
            images_path_train.append(path_train + dirname)

for root, dirs, files in os.walk(path_test):
    for dirname in sorted(files):
        splitname = dirname.split('.')
        if len(splitname) > 1:
            images_path_test.append(path_test + dirname)


def build_data(image):
    image_array = np.hstack(np.array(Image.open(image).convert('L')))
    return image_array


for image in images_path_train:
    images_train.append(build_data(image))
for image in images_path_test:
    images_test.append(build_data(image))

X_train = np.array(images_train)
X_test = np.array(images_test)

################################################
direction_encode = {'right': 0, 'left': 1, 'up': 2, 'straight': 3}
labels_train = []
labels_test = []


def build_label(image):
    direction = image.split('/')[-1].split('_')[1]
    return direction_encode[direction]


for image in images_path_train:
    labels_train.append(build_label(image))

for image in images_path_test:
    labels_test.append(build_label(image))

y_train_directionfaced = np.array(labels_train)
y_test_directionfaced = np.array(labels_test)


#################################################
# part a
import time


start = time.time()


rnd_clf	=	RandomForestClassifier(random_state=0)
rnd_clf.fit(X_train,y_train_directionfaced)


stop=time.time()
time=stop-start
y_pred_rnd = rnd_clf.predict(X_test)
accuracy=accuracy_score(y_test_directionfaced,y_pred_rnd)
print 'part a :',accuracy,time
joblib.dump([rnd_clf,time,accuracy], 'part_a.pkl', protocol=2)


####################################################
# part b:
import time

pca = PCA(n_components=0.95)

X_reduced_train = pca.fit_transform(X_train)

start_=time.time()


rnd_clf_new	=	RandomForestClassifier(random_state=0)
rnd_clf_new.fit(X_reduced_train,y_train_directionfaced)

stop_=time.time()

time_=stop_- start_
X_test_reduced = pca.transform(X_test)
y_pred_rnd_pca = rnd_clf_new.predict(X_test_reduced)
accuracy_pca=accuracy_score(y_test_directionfaced,y_pred_rnd_pca)

print 'part b:',accuracy_pca,time_
joblib.dump([rnd_clf_new,time_,accuracy_pca], 'part_b.pkl', protocol=2)

########################################################
emotion_encode = {'neutral': 0, 'happy': 1, 'angry': 2, 'sad': 3}
labels_train_emotions = []
labels_test_emotions = []


def build_label_emotions(image):
    direction = image.split('/')[-1].split('_')[2]
    return emotion_encode[direction]


for image in images_path_train:
    labels_train_emotions.append(build_label_emotions(image))

for image in images_path_test:
    labels_test_emotions.append(build_label_emotions(image))

y_train_emotion = np.array(labels_train_emotions)
t_test_emotion = np.array(labels_test_emotions)

########################################################

# part c
import time

start_1=time.time()
log_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=0)
log_reg.fit(X_train, y_train_emotion)
stop_1=time.time()

time_1=stop_1- start_1
y_pred_log = log_reg.predict(X_test)
accuracy_log=accuracy_score(t_test_emotion, y_pred_log)
print 'part c:', accuracy_log,time_1
joblib.dump([log_reg,time_1,accuracy_log], 'part_c.pkl', protocol=2)

#########################################################
#part d
import time

start_2=time.time()


log_reg2 = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=0)
log_reg2.fit(X_reduced_train,y_train_emotion)

stop_2=time.time()

time_2=stop_2- start_2
y_pred_log_pca = log_reg2.predict(X_test_reduced)
accuracy_pca_log=accuracy_score(t_test_emotion,y_pred_log_pca)
print 'part d:',accuracy_pca_log,time_2
joblib.dump([log_reg2,time_2,accuracy_pca_log], 'part_d.pkl', protocol=2)
