#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np


# In[ ]:


from google.colab import drive
drive.mount('/content/drive/')


# In[ ]:


from scipy.io import loadmat
train_data1 = loadmat('Vol_01_input.mat')
train_data2 = loadmat('Vol_02_input.mat')
train_data3 = loadmat('Vol_05_input.mat')


# In[ ]:


val_data = loadmat('Vol_06_input.mat')


# In[ ]:


train_label1 = loadmat('Vol_01gt.mat')
train_label2 = loadmat('Vol_02gt.mat')
train_label3 = loadmat('Vol_05gt.mat')
# for keys in train_label1:
#   print(keys, train_label1[keys])


# In[ ]:


val_gt = loadmat('Vol_06gt.mat')


# In[ ]:


len(train_data1)


# In[ ]:


tr_data = train_data1['ana']
tr_data2 = train_data2['ana']
tr_data3 = train_data3['ana']
print(tr_data.shape[0])


# In[ ]:


vl_data = val_data['ana']
vl_gt = val_gt['gt']


# In[ ]:


tr_gt = train_label1['gt']
tr_gt2 = train_label2['gt']
tr_gt3 = train_label3['gt']


# In[ ]:


train_data = []
for i in range(tr_data.shape[2]):
  train_data.append(tr_data[:,][:,][i])
for i in range(tr_data.shape[2]):
  train_data.append(tr_data2[:,][:,][i])
for i in range(tr_data.shape[2]):
  train_data.append(tr_data3[:,][:,][i])
print(len(train_data))
print(train_data[0].shape)


# In[ ]:


val_data = []
for i in range(vl_data.shape[2]):
  val_data.append(vl_data[:,][:,][i])
val_gt = []
for i in range(vl_gt.shape[2]):
  val_gt.append(vl_gt[:,][:,][i])
print(len(val_data), val_data[0].shape)


# In[ ]:


train_gt = []
for i in range(tr_gt.shape[2]):
  train_gt.append(tr_gt[:,][:,][i])
for i in range(tr_gt.shape[2]):
  train_gt.append(tr_gt2[:,][:,][i])
for i in range(tr_gt.shape[2]):
  train_gt.append(tr_gt3[:,][:,][i])
print(len(train_gt))
print(train_gt[0].shape)


# In[ ]:


def check(mat, st_i, st_j):
  i = st_i
  j = st_j
  flag = 0
  # print(i, j)
  while(i < st_i + 32):
    j = st_j
    while(j < st_j + 32):
      if mat[i][j] == 0:
        return False
      # if mat[i][j] == 1:
      #   flag = 1
      j += 1
    i += 1
  # if flag == 1:
  #   return True
  # else:
  #   return False
  return True

def make_patch(mat, st_i, st_j):
  patch = np.zeros((32, 32))
  i = st_i - 16
  j = st_j - 16
  a = 0
  b = 0
  if i < 0:
    i = st_i
    st_i = 16
  if j < 0:
    j = st_j
    st_j = 16
    
  while(i < st_i + 16):
    j = st_j - 16
    b = 0
    while(j < st_j  + 16):
      patch[a][b] = mat[i][j]
      j += 1
      b += 1
    i += 1
    a += 1
  return patch


jump = 8
new_train_data = []
new_train_gt = []
for k in range(len(train_gt)):
  for i in range(tr_gt.shape[1] - 32):
    flag = 0
    for j in range(tr_gt.shape[2] - 32):
      
      if train_gt[k][i][j] == 1:
        allowed = check(train_gt[k], i, j)
        if allowed == True:
          patch_data = make_patch(train_data[k], i, j)
          patch_gt = make_patch(train_gt[k], i, j)
          new_train_data.append(patch_data)
          patch_gt = patch_gt - 1
          new_train_gt.append(patch_gt)
          j = j + jump
          i = i + jump
          # flag = 1
        # elif allowed == False and index >= 0:
        #   j = index
    if flag == 1:
      i = i + jump
print(len(new_train_data), len(new_train_gt))
print(new_train_data[12].shape, new_train_gt[12].shape)


# In[ ]:


np.save('new_train_data_v6.npy', new_train_data)
np.save('new_train_gt_v6.npy', new_train_gt)


# In[ ]:


print(train_gt[0] == 1)
def check(mat, st_i, st_j):
  i = st_i
  j = st_j
  flag = 0
  # print(i, j)
  while(i < st_i + 32):
    j = st_j
    while(j < st_j + 32):
      if mat[i][j] == 0:
        return False
      if mat[i][j] == 1:
        flag = 1
      j += 1
    i += 1
  # if flag == 1:
  #   return True
  # else:
  return True

def make_patch(mat, st_i, st_j):
  patch = np.zeros((32, 32))
  i = st_i
  j = st_j
  a = 0
  b = 0
  while(i < st_i + 32):
    j = st_j
    b = 0
    while(j < st_j + 32):
      patch[a][b] = mat[i][j]
      j += 1
      b += 1
    i += 1
    a += 1
  return patch

jump = 30
new_train_data = []
new_train_gt = []
for k in range(len(train_gt)):
  for i in range(tr_gt.shape[1] - 32):
    flag = 0
    for j in range(tr_gt.shape[2] - 32):
      if i != 0 and train_gt[k][i - 1][j] == 1 and train_gt[k][i - 30][j] != 1:
        continue
      if train_gt[k][i][j] == 1:
        if check(train_gt[k], i, j) == True:
          patch_data = make_patch(train_data[k], i, j)
          patch_gt = make_patch(train_gt[k], i, j)
          new_train_data.append(patch_data)
          patch_gt = patch_gt - 1
          new_train_gt.append(patch_gt)
          j = j + jump
          # i = i + 2
          flag = 1
    # if flag == 1:
    #   i = i + jump
          

print(len(new_train_data), len(new_train_gt))
print(new_train_data[12].shape, new_train_gt[12].shape)


# In[ ]:


np.save('new_train_data.npy', new_train_data)
np.save('new_train_gt.npy', new_train_gt)

