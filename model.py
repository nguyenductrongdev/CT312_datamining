# -*- coding: utf-8 -*-

# Nạp các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import KFold
import pickle

# Đọc dữ liệu từ file excel
rice_df = pd.read_excel("Rice_Osmancik_Cammeo_Dataset.xlsx")
# rice_df.describe()

# Tách nhãn và các thuộc tính từ dataset
data = rice_df.iloc[:, :-1]
target = rice_df.iloc[:, -1]
# print(data.shape)
# print(target.shape)

# Scaling dữ liệu với phương pháp standardlization
scaler = MinMaxScaler()

# Hàm đánh giá


def kfold(model):
    recall = []
    kf = KFold(n_splits=10, shuffle=True, random_state=100)
    run = 0
    # Khởi tạo các list dùng để lưu trữ accuracy, precision, recall qua các lần lặp
    accuracy = []
    precision = []
    recall = []
    for train_index, test_index in kf.split(data):
        run += 1
        # Tách x, y từ tập train và tập test
        data_train, data_test = data.iloc[train_index,
                                          :], data.iloc[test_index, :]
        target_train, target_test = target.iloc[train_index], target.iloc[test_index]
        scaler.fit_transform(data_train)
        scaler.transform(data_test)
        # Huấn luyện mô hình
        model.fit(data_train, target_train)
        # Dự đoán mô hình
        predict = model.predict(data_test)
        # Đánh giá bằng accuracy score
        acc = accuracy_score(target_test, predict)
        accuracy.append(acc)
        # Tính toán precision score
        pre = precision_score(target_test, predict, labels=['Cammeo', 'Osmancik'],
                              average=None, zero_division=0)
        precision.append(pre)
        # Tính toán recall score
        rec = recall_score(target_test, predict, average=None, labels=[
                           'Cammeo', 'Osmancik'], zero_division=0)
        recall.append(rec)
        # print("----------------------------------------")
        # print('run ', run,)
        # # print("TRAIN:", train_index, "TEST:", test_index)
        # print(target_test.value_counts())
        # print('accuracy: ', acc)
        # print('precision', pre)
        # print('recall: ', rec)

    labels = ['Cammeo', 'Osmancik']
    # Tính accuracy trung bình
    accuracy = np.mean(accuracy)
    # Tính precision trung bình
    precision = np.array(precision).mean(axis=0)
    precision = dict(zip(labels, precision))
    # Tính recall trung bình
    recall = np.array(recall).mean(axis=0)
    recall = dict(zip(labels, recall))

    # print(precision)
    # print(recall)

    return {'accuracy': accuracy,
            'precision': precision,
            'recall': recall}


# ========================== Tạo mô hình Naive Bayes
nb_model = GaussianNB()
nb_result = kfold(nb_model)
print('\n------naive Bayes---------')
print(nb_result)
# KL:
#   Accuracy = 91.05
#   Precision (Cameo, Omancik) = 90.12, 91.79
#   Recall (Cameo, Omancik) = 88.84, 92.64


# ========================== SVM
kernels = ["linear", "poly", "rbf", "sigmoid"]
svm_results = dict()
# Thử xây dựng mô hình với các kernel khác nhau
for kernel in kernels:
    svm_model = SVC(kernel=kernel)
    svm_results[kernel] = kfold(svm_model)

# Tìm kernel có độ chính xác tốt nhất
print('\n------SVM---------')
# print(svm_results)
best_acc = 0
best_kernel = ""
# Lặp qua lần lượt các Kernel để tìm ra kerel mang lại accuracy tốt nhất
for kernel in kernels:
    if svm_results[kernel]['accuracy'] > best_acc:
        best_acc = svm_results[kernel]['accuracy']
        best_kernel = kernel
# Hiển thị kernal mang lại accuracy tốt nhất và precision, recall nếu xây dựng svm model với kernel đó
print('best_kernel: ', best_kernel, ' -- best_acc: ', best_acc,
      '--precision: ', svm_results[best_kernel]['precision'],
      '--recall: ', svm_results[best_kernel]['recall'])
# KL:
#   accuracy = 93.10
#   Precision (Cameo, Omancik) = 92.14, 93.78
#   Recall (Cameo, Omancik) = 91.64, 94.20

# Decision Tree
dt_result = dict()
# Xây dựng mô hình với max_depth từ 3 - 20 để tìm ra độ sâu tốt nhất
for depth in range(2, 20):
    dt_model = DecisionTreeClassifier(
        criterion="entropy", max_depth=depth,  random_state=100)
    dt_result[depth] = kfold(dt_model)

# Tìm độ sâu tốt nhất cho cây quyết định
print('\n------Decision Tree---------',)
# print(dt_result)
best_acc = 0
best_depth = -1
# Tìm độ sâu trong đoạn [2, 19] để tìm ra độ sâu mang lại độ chính xác cao nhất
for i in range(2, 20):
    if dt_result[i]['accuracy'] > best_acc:
        best_acc = dt_result[i]['accuracy']
        best_depth = i
# Hiển thị độ sâu mang lại accuracy tốt nhất và precision, recall khi xây dựng cây với độ sâu đó
print('best_depth: ', best_depth, ' -- best_acc: ', best_acc,
      '--precision: ', dt_result[best_depth]['precision'],
      '--recall: ', dt_result[best_depth]['recall'])
# KL:
#   accuracy = 92.36
#   Precision (Cameo, Omancik) = 91.66, 92.97
#   Recall (Cameo, Omancik) = 90.33, 93.77

# Vẽ cây quyết định để quan sát và lưu lại
clf = DecisionTreeClassifier(criterion='entropy')
fig, ax = plt.subplots(figsize=(200, 100))
plot_tree(clf.fit(scaler.fit_transform(data), target),
          feature_names=rice_df.columns[:-1],
          class_names=['Cammeo', 'Osmancik'], filled=True)
plt.savefig('tree.png')
print(clf.tree_.max_depth)
# Kết quả: Mô hình SVM với kernel linear có độ chính xác cao nhất nên lưu lại mô hình để dự đoán
scaler.fit(data)
data = scaler.transform(data)
model = SVC(kernel='linear')
model.fit(data, target)
model.scaler = scaler
path = './riceclassify/static/model.sav'
pickle.dump(model, open(path, 'wb'))
