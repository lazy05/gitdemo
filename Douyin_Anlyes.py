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
    for line in reader:
        test_x.append(line[1:7])
        test_y.append(line[-1])
test_x = np.array(test_x).astype(int)
test_y = np.array(test_y).astype(int)
print(test_x.shape)
print(test_y.shape)

gender = test_x[:, 4]
print(gender)
dic1 = {}
for i in range(0, 3):
    dic1[i] = 0
    for j in gender:
        if i == j:
            dic1[i] += 1
lt = [dic1[0], dic1[1], dic1[2]]
mp.figure('Gender')
mp.title('Gender', fontsize=20)
mp.pie(lt, [0.01, 0.01, 0.05], ['None', 'boy', 'girl'],
       autopct='%1.2f%%', shadow=True, startangle=90)
mp.axis('equal')
mp.show()

a, b, c, d = 0, 0, 0, 0
for i in test_y:
    print(i)
    if i <= 10000:
        a += 1
    elif 10000 < i <= 100000:
        b += 1
    elif 100000 < i <= 1000000:
        c += 1
    else:
        d += 1
mp.figure('Fans')
mp.title('Fans', fontsize=20)
mp.pie([a, b, c, d], [0.01, 0.01, 0.01, 0.05], ['1W', '1W-10W',
                                                '10W-100W', '100W', ],

       autopct='%1.2f%%', labeldistance=1.1, shadow=True, startangle=90, pctdistance=0.8)
mp.axis('equal')
mp.show()
