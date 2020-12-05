# -*- coding: utf-8 -*-

# Nap cac thu vien can thiet
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
# Doc du lieu tu file excel
rice_df = pd.read_excel("Rice_Osmancik_Cammeo_Dataset.xlsx")
# rice_df.describe()

# Tách nhãn và các thuộc tính từ dataset
data = rice_df.iloc[:, :-1]
target = rice_df.iloc[:, -1]

# Scaling dữ liệu với phương pháp standardlization
scaler = MinMaxScaler()

# --------------------- Xay dung mo hinh Naive Bayes voi phan phoi Gauss
nb_model = GaussianNB()
pipeline = Pipeline([('transformer', scaler), ('estimator', nb_model)])
nb_score = cross_val_score(pipeline, data, target, cv=15)
nb_score_my = np.mean(nb_score)
print('Do chinh xac trung binh NB:', nb_score_my)

# KL: Naive Bayes co do chinh xac la 0.905

# # Luu lai Naive Bayes model
# NaiveBayesModel = './riceclassify/static/naive_bayes_model.sav'
# pickle.dump(nb_model, open(NaiveBayesModel, 'wb'))


# --------------------- Xay dung mo hinh SVM
kernel = ["linear", "poly", "rbf", "sigmoid"]
svm_model = []
svm_scores = dict()
# Thu xay dung mo hinh voi kernel khac nhau
for k in kernel:
    m = SVC(kernel=k)
    svm_model.append(m)
    pipeline = Pipeline([('transformer', scaler), ('estimator', m)])
    score = cross_val_score(pipeline, data, target, cv=15)
    svm_scores[k] = np.mean(score)

# Tim kernel co do chinh xac tot nhat
val = list(svm_scores.values())
key = list(svm_scores.keys())
best_acc = max(val)
best_kernel = key[val.index(best_acc)]

print("SVM: ", best_kernel, best_acc)
# KL: SVM co do chinh xac la 0.9288713910761153 voi kernel la poly


# --------------------- Xay dung mo hinh cay quyet dinh
dt_score = dict()
dt_model = []
# Xay dung mo hinh voi max_depth tu 3 - 20 de tim ra do sau tot nhat
for depth in range(2, 8):
    clf = DecisionTreeClassifier(
        criterion="entropy", max_depth=depth,  random_state=100)
    dt_model.append(clf)
    pipeline = Pipeline([('transformer', scaler), ('estimator', clf)])
    scores = cross_val_score(
        clf, data, target, cv=15)
    dt_score[depth] = np.mean(scores)

# print(dt_score)
val = list(dt_score.values())
key = list(dt_score.keys())
best_acc = max(val)
best_depth = key[val.index(best_acc)]
print('Do sau - do chinh xac DT: ', best_depth, best_acc)
# KQ: max_depth = 5 dem lai do chinh xac cao nhat (0.9236220472440944)

# ==> Mô hình SVM có độ chính xác cao nhất
# ===> Lưu lại SVM làm mô hình dự đoán

scaler.fit(data)
data = scaler.transform(data)
model = dt_model[val.index(best_acc)].fit(data, target)
model.scaler = scaler
path = './riceclassify/static/model.sav'
pickle.dump(model, open(path, 'wb'))
