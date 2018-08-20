# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import csv
import numpy as np
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp
test_x, test_y = [], []
with open('LZ/douyin2.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        # print(row)
        if row[0]:
            year = row[0].split('/')
            if len(year) == 3:
                num2 = row[1:6]
                if 0 < (2018 - int(year[0])) <= 50:
                    year = [2018 - int(year[0])]
                    test_x.append(year + num2)
                    test_y.append(row[-2])

test_x = np.array(test_x).astype(int)
test_y = np.array(test_y).astype(int)
print(test_x.shape)
print(test_y.shape)


x = test_x[:, 0]
dic = {}
for i in range(1, 51):
    dic[i] = 0
    for j in x:
        if i == j:
            dic[i] += 1
bar_x = [i for i in dic]
bar_y = []
for k in bar_x:
    bar_y.append(dic[k])
mp.figure('Age')
mp.title('Age', fontsize=20)
mp.xlabel('age', fontsize=12)
mp.ylabel('sum', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(axis='y', linestyle=':')
mp.bar(bar_x, bar_y, facecolor='deepskyblue', edgecolor='steelblue')
mp.show()

x, y = su.shuffle(test_x, test_y, random_state=15)
train_size = int(len(x) * 0.9)
train_x, test_x, train_y, test_y =\
    x[:train_size], x[train_size:],\
    y[:train_size], y[train_size:]
model = se.RandomForestRegressor(max_depth=10,
                                 n_estimators=300, min_samples_split=2)
model.fit(train_x, train_y)
fi_dy = model.feature_importances_
print(fi_dy)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))

mp.figure('Feature Importance')
mp.title('Douyin', fontsize=16)
mp.ylabel('Importance', fontsize=12)
mp.tick_params(labelsize=10)
mp.grid(axis='y', linestyle=':')
mp.bar(['birthday', 'dynamic', 'constellation', 'favoriting_count', 'following_count',
        'gender'], fi_dy, width=0.5,
       facecolor='lightcoral',
       edgecolor='indianred')
mp.xticks(['birthday', 'dynamic', 'constellation', 'favoriting_count', 'following_count',
           'gender'], rotation=20)
mp.tight_layout()
mp.show()
