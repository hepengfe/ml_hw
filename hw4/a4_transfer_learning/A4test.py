#%%

import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import copy
#%%
def train(model, train_loader, val_loader, split , gpu, lr = 1e-4, num_epochs = 20, criterion = nn.CrossEntropyLoss()):
    from tqdm import tqdm
    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
    # model = torch.nn.DataParallel(model).cuda(0)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    log_train_loss = []
    log_val_loss = []
    log_val_acc = []
    for epoch in tqdm(range(num_epochs)):
        # training
        model.train()  # if using batchnorm or dropout, use train mode setting! don't want to adjust normalization on non-train data
        epoch_train_loss = 0
        for batch_idx, (input, target) in enumerate(train_loader):
            input = input.cuda(gpu, non_blocking=True)  # move to device and zero optimizer
            target = target.cuda(gpu, non_blocking=True)
            optimizer.zero_grad()
            ### train step ###
            output = model(input)  # forward
            loss = criterion(output, target)
            ### end train step ###
            ### backward pass and optim step ###
            loss.backward()
            optimizer.step()
            ### logging
            epoch_train_loss += loss
        log_train_loss.append(epoch_train_loss / (batch_idx + 1))

        # evaluation
        model.eval()  # set batchnorm + dropout in eval so it doesn't adjust on validation data
        with torch.no_grad():  # turn off gradients
            epoch_val_loss = 0
            num_correct = 0
            highest_val_acc = 0
            highest_val_acc_model = None
            for batch_idx, (input, target) in enumerate(val_loader):
                # do the same steps for train step as for val step but skip updates and backward pass (no gradients)
                input = input.cuda(gpu, non_blocking=True)
                target = target.cuda(gpu, non_blocking=True)
                # log val loss every val step
                output = model(input)
                loss = criterion(output, target)
                epoch_val_loss += loss
                # validation accuracy
                num_correct_per_batch = torch.sum(target == torch.argmax(output, axis=1))
                num_correct += num_correct_per_batch
            val_accuracy = num_correct.item() / split
            log_val_acc.append(val_accuracy)
            print("validation accuracy: ", val_accuracy)
            if val_accuracy > highest_val_acc:
                highest_val_acc = val_accuracy
                highest_val_acc_model = copy.deepcopy(model)
            log_val_loss.append(epoch_val_loss / (batch_idx + 1))  # average the loss
    return highest_val_acc_model, log_train_loss, log_val_loss, log_val_acc

def import_dataset():
    transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    train = torchvision.datasets.CIFAR10(".", train = True, transform = transform)
    val = torchvision.datasets.CIFAR10(".", train = True, transform = transform)
    test = torchvision.datasets.CIFAR10(".", train = False, transform = transform)
    return train, val, test
#%% Data loader

def fixed_feature_alexnet():
    # model
    model = models.alexnet(pretrained=True)
    for param in model.parameters():
        param.require_grad = False
    model.classifier[6] = nn.Linear(4096, 10)
    return model

#%%


#%% Train vs Validation
def eval_test_acc(model, M,  test_loader, gpu):
    num_correct = 0
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(test_loader):
            input = input.cuda(gpu, non_blocking= True)
            target = target.cuda(gpu, non_blocking= True)
            output = model(input)
            num_correct_per_batch = torch.sum(target == torch.argmax(output, axis=1))
            num_correct += num_correct_per_batch
    return num_correct.item()/M

def report(name, loss1, loss2, val_acc_l, test_acc):
    plt.plot(loss1, label = "Train Loss")
    plt.plot(loss2, label = "Validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(name)
    fname = name + ".txt"
    with open(fname, "w") as f:
        txt = "Highest validation accuracy: " + str(max(val_acc_l)) + "\n"
        txt += "Final test accuracy: " +  str(test_acc)
        f.write(txt)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=int, default=1, help="1 is transfer learning, 2 is fine tuning")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()


    CIFAR10_train, CIFAR10_val, CIFAR10_test = import_dataset()

    val_ratio = 0.1
    N = len(CIFAR10_train)
    M = len(CIFAR10_test)
    np.random.seed(10)
    idx = np.random.randint(0, N, size = N)
    split = int(N * val_ratio)
    train_idx, val_idx = idx[split:], idx[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    num_of_classes = 10
    lr = 0.05
    batch_size = 32
    criterion = nn.CrossEntropyLoss()

    val_ratio = 0.1 #
    train_loader = torch.utils.data.DataLoader(CIFAR10_train, batch_size = batch_size, sampler = train_sampler)
    val_loader = torch.utils.data.DataLoader(CIFAR10_train, batch_size = batch_size, sampler = val_sampler)
    test_loader = torch.utils.data.DataLoader(CIFAR10_test, batch_size = 100)

    if args.model == 1:
        model1 = fixed_feature_alexnet()
        model1, log_train_loss1, log_val_loss1, log_val_acc1 = train(model1, train_loader, val_loader, split, gpu = args.gpu, num_epochs= args.num_epochs)
        #%%
        test_acc1 = eval_test_acc(model1, M, test_loader, gpu = args.gpu)
        report("A4a Transfer Learning", log_train_loss1, log_val_loss1, log_val_acc1, test_acc1)
    elif args.model == 2:
        #%% Model2: Fine tuning
        model2 = models.alexnet(pretrained=True)
        model2.classifier[6] = nn.Linear(4096, 10)
        model2, log_train_loss2, log_val_loss2, log_val_acc2 = train(model2, train_loader, val_loader, split, gpu = args.gpu, num_epochs= args.num_epochs)
        test_acc2 = eval_test_acc(model2, M, test_loader, gpu = args.gpu)
        report("A4b fine tuning", log_train_loss2, log_val_loss2, log_val_acc2, test_acc2)
    else:
        print("wrong model argument, it's either 1 or 2")

