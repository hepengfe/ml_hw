#!/usr/bin/env python
# coding: utf-8

# In[101]:


from mnist import MNIST
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


# In[102]:


def load_dataset():
    mndata = MNIST("./data/python-mnist/data/")
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, labels_train, X_test, labels_test
X_train, labels_train, X_test, labels_test = load_dataset()
X_train = torch.from_numpy(X_train).float().cuda()
y_train = torch.from_numpy(labels_train).long().cuda()
X_test = torch.from_numpy(X_test).float().cuda()
y_test = torch.from_numpy(labels_test).long().cuda()


# In[124]:


n, d = X_train.shape
m = X_test.shape[0]


# In[125]:


def train(X_train, y_train, X_test, y_test, reg = None, step_size = 0.01, stop = 1e-4):
    class_rate_train_l = []
    class_rate_test_l = []
    epochs = 50
    W = torch.zeros(784, 10, device = 0 , requires_grad= True).float()
#     step_size = 0.01
    n = y_train.shape[0]
    k = 10 # number of classes
    y_train_onehot = torch.FloatTensor(n, k).zero_().cuda()
    y_train_onehot.scatter_(1, y_train.unsqueeze(-1).long(), 1)
    W_update = torch.tensor([1])
    with torch.cuda.device(0):
        while W_update > stop: # TODO: converge condition
            if reg == "ridge":
                y_hat = torch.matmul(X_train, W)
                criterion = torch.nn.MSELoss()
                loss = criterion(y_hat, y_train_onehot)
            elif reg == "logistics":
                y_hat = torch.matmul(X_train, W)
                loss = F.cross_entropy(y_hat, y_train)
            else:
                assert reg == None, "needs regression type"

            
            loss.backward()
            W.data = W.data - step_size * W.grad
            W_update = torch.norm(step_size * W.grad, p = 2)

            y_pred_train =  torch.argmax(torch.matmul(X_train, torch.matmul(W,  torch.eye(10).cuda())), axis = 1)
            y_pred_test = torch.argmax(torch.matmul(X_test, torch.matmul(W,  torch.eye(10).cuda())), axis = 1)
            class_rate_train = torch.sum(y_pred_train ==y_train).item()/n
            class_rate_test = torch.sum(y_pred_test == y_test).item()/m
            class_rate_train_l.append(class_rate_train)
            class_rate_test_l.append(class_rate_test)
    #         print("W.grad: ", W.grad)
            print("W_update: ", W_update)


            W.grad.zero_()
    return class_rate_train_l, class_rate_test_l


# In[126]:


class_rate_train_l1, class_rate_test_l1 =    train(X_train, y_train, X_test, y_test, reg = "ridge")


# In[127]:


class_rate_train_l2, class_rate_test_l2 =    train(X_train, y_train, X_test, y_test, reg = "logistics")


# In[128]:


class_rate_train_l1


# In[130]:


xs = range(len(mis_class_rate_train_l1))
plt.plot(xs, class_rate_train_l1, label = "train")
plt.plot(xs, class_rate_test_l1, label= "test")
plt.legend()
plt.xlabel("number of iterations")
plt.ylabel("accuracy")
plt.title("ridge regression")
plt.savefig("B4c1")
plt.show()


xs = range(len(mis_class_rate_train_l2))
plt.plot(xs, class_rate_train_l2, label = "train")
plt.plot(xs, class_rate_test_l2, label= "test")
plt.xlabel("number of iterations")
plt.ylabel("accuracy")
plt.legend()
plt.title("logistics regression")
plt.savefig("B4c2")
plt.show()

