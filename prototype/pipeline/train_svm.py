from sklearn import svm
import csv
import pickle
from sklearn.utils import shuffle
import os as _os

_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_SCRIPT_DIR)
dis_path = _os.path.join(_ROOT, 'pipeline') + '/'
dat_path = _os.path.join(_ROOT, 'models', 'emotion') + '/'

dat=[] #Список, содержащий дискрипторы
emotion_number=[] #Список, содержащий номера эмоций
file_names=['Em_1_128.txt','Em_2_128.txt','Em_3_128.txt','Em_4_128.txt','Em_5_128.txt','Em_6_128.txt','Em_7_128.txt']
for i, file_name in enumerate(file_names):
    f=open(dis_path+file_name,'r')
    reader = csv.reader(f)
    for row in reader:
        dat.append(row)
        emotion_number.append(i+1)
    f.close

pp=int(len(dat)*0.8)
dat, emotion_number = shuffle(dat, emotion_number, random_state=0)
learning_dat,learning_emotion_number=dat[:pp], emotion_number[:pp]
test_dat,test_emotion_number=dat[pp+1:], emotion_number[pp+1:]

clf = svm.SVC(C=2.0, gamma=2.0 , kernel='poly', degree=3)
clf.fit(learning_dat,learning_emotion_number)
succesfull_prediction = clf.score(test_dat,test_emotion_number)
print(f'Точность SVM: {succesfull_prediction:.2%}')

f=open(dat_path+'svm_dat.dat','wb')
pickle.dump(clf,f)
f.close
print(f'Модель сохранена: {dat_path}svm_dat.dat')
