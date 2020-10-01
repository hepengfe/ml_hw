#!/usr/bin/env python
# coding: utf-8

# In[9]:


import sys
import numpy as np


# In[10]:


# TODO: is the 0 weights random or just with index larger than k
n = 500
d = 1000
k = 100
sigma = 1


train_x = np.random.normal(0,1, (n,d))


# use two np arange?
non_empty_weights = np.arange(k)
# w = np.zeros(d) # initial weights ready for training
true_w = np.zeros(d)
true_w[non_empty_weights] = 1  # it's only for generating data
w = np.zeros(d)
print(w.shape)

train_y = train_x.dot(true_w) + np.random.normal(0,1, n)

print(train_x)
print(train_y)


# In[31]:


a = np.array([0.0])

np.count_nonzero(a)


# In[48]:


# TODO: regularization path
# it updates values column by column and each time use all data


reg_lambda_l = []
# initialize lambda
for k in range(d):
    reg_lambda = 2*np.abs(np.sum(train_x[:,k] *(train_y-np.average(train_y))   ,axis=0))
    reg_lambda_l.append(reg_lambda)
reg_lambda = max(reg_lambda_l)

print("max_lambda: ", reg_lambda)


num_nonzero_l = []
reg_lambda_l = []
reg_lambda_l.append(reg_lambda)

objective_val_l = []

FDR_l = []
TPR_l = []
true_k = 100


w_updates = [np.zeros(d)] # initialize how much w_updates
w_prev = np.zeros(d)
b_l = [(1/n)*np.sum( train_y - train_x.dot(w))]
while reg_lambda > 1e-2:  # TODO
    max_update = sys.maxsize
#     # for each lambda solution, compute w from zero vector
#     w_prev = np.zeros(d)
    w = w_updates[-1]
    w = w_prev
    b = b_l[-1]
    
    count = 0
    
    while max_update >= 0.05: # condition for not converge
        # pre-compute a
        # as a is fixed and solely depends on X
        a = 2*np.sum(train_x**2, axis = 0)  # axis = 0 by default
    #     print(a.shape)
        for k in range(d):
            excluded_train_x = np.delete(train_x, k, axis = 1)
            excluded_w = np.delete(w, k)
            # calculate the cost
            c_k = 2*np.sum(train_x[:,k]*(train_y- (b+ excluded_train_x.dot(excluded_w))  )  )
            # update weights
            if c_k < -reg_lambda:
                w[k]= (c_k+reg_lambda)/a[k]
            elif c_k > reg_lambda:
                w[k] = (c_k- reg_lambda)/a[k]
            else:
                w[k] = 0
            
        # sanity check
        obj_val = np.sum(  (train_x.dot(w) + b - train_y)**2 )+ np.sum(reg_lambda * np.absolute(w))
        objective_val_l.append(obj_val)
        
        w_updates.append(np.array(w))  # append the updated w
        b = (1/n)*np.sum( train_y - train_x.dot(w))
        b_l.append(b)

        if len(w_updates) > 2:  # TODO: maximum 
            print("w_updates ", len(w_updates))
            max_update = np.max(w_updates[-1]-w_updates[-2])
            print("maximum update ",  max_update)
            if max_update == 0: # skip zero weight matrix
                break
        w_prev = w
#     correct_num_nonzero = np.sum((w != 0) * (true_w != 0))
    k = 100
    correct_num_nonzero = np.sum( w[:k] != 0)
    
#     print("w: ", w)
#     print("true_w", true_w)
    print("correct_num_nonzero: ", correct_num_nonzero)
    count += 1
    if count >= 10:
        break
    
    incor_num_nonzero = np.sum( w[k:] != 0 )
    
    # keep track of number of nonzero for each solution
    num_nonzero = np.sum(w != 0)
    num_nonzero_l.append(num_nonzero)
    # FDR
    if num_nonzero == 0:
        # invalid number of nonzero
        FDR_l.append(0.0)
    else:
        FDR_l.append(incor_num_nonzero/num_nonzero)
    # TPR
    TPR_l.append(correct_num_nonzero/true_k)
    
#     print("FDR_l ", FDR_l)
#     print("TPR_l: ", TPR_l)
#     break
    
    # regularization lambda
    reg_lambda = reg_lambda*0.5
    reg_lambda_l.append(reg_lambda)
    print("new lambda: ", reg_lambda)


# In[13]:


# sanity check continue
count = 0
for i in range(1,len(objective_val_l)):
    if objective_val_l[i-1] < objective_val_l[i]:
        count += 1
print("number of wrong optimization: " , count)
print(objective_val_l)


# In[22]:


from matplotlib import pyplot as plt
reg_lambda_0 = max(reg_lambda_l)
plt.plot(reg_lambda_l[:-1], num_nonzero_l)
plt.xscale("log")
# plt.xlim(reg_lambda_0, 1e-2) 
plt.xlim(1e-2, reg_lambda_0) 
plt.xlabel("reg_lambda")
plt.ylabel("number of nonzero")
plt.savefig("A4a")
plt.show()


# In[42]:


len(w_updates)


# In[16]:


print(reg_lambda_l)


# In[45]:


print(FDR_l)

print(TPR_l)


# In[49]:


plt.plot(FDR_l, TPR_l)
plt.xlabel("FDR")
plt.ylabel("TPR")
plt.savefig("A4b")
plt.show()

