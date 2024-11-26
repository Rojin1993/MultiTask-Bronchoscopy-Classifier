# --------------------------
# Import necessary libraries
# --------------------------
import copy
import time
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, SubsetRandomSampler
import torchvision
import OnePartSeDensenet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
import os

# --------------------------
# Seed initialization
# --------------------------
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------------
# Configuration and constants
# --------------------------
train_b_s = 14
valid_b_s = 10
test_b_s = 5
valid_size = 50
num_epochs = 20
k = 5  # Number of folds for cross-validation
d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes1 = ["Normal", "AbNormal"]
classes2 = ["Tuberclosis", "Cancer"]
classes3 = ['Normal', 'Tuberclosis', 'Cancer']
PATH = 'R:/Bronchoscopy' # Path of saving final weights of each fold

def train_epoch(model, de, dataloader, loss_fn1, loss_fn2, optimizer):
    train_loss1, train_correct1 = 0.0, 0
    train_loss2, train_correct2 = 0.0, 0
    train_correct = 0
    numberAbnormal = 0

    predictions2_total_changed = []
    predictions_total = []
    labels_total = []
    predictions1_total = []
    labels1_total = []
    predictions2_total = []
    labels2_total = []

    model.train()

    for images, labels in dataloader:
        images = images.to(de)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(de)

        for l in labels:
            labels_total.append(l)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            output1, output2 = model(images)

            # Classifier of normal and abnormal
            labels1 = []
            c = 0
            while c < len(labels):
                if labels[c] == 2 or labels[c] == 1:
                    labels1.append(1)
                else:
                    labels1.append(0)
                c += 1

            labels1 = torch.Tensor(labels1)
            labels1 = labels1.type(torch.LongTensor)
            labels1 = labels1.to(de)

            loss1 = loss_fn1(output1, labels1)
            scores1, predictions1 = torch.max(output1.data, 1)

            train_correct1 += (predictions1 == labels1).sum().item()
            train_loss1 += loss1.item() * images.size(0)

            for lab in labels1:
                labels1_total.append(lab)
            for pr in predictions1:
                predictions1_total.append(pr)

            # Classifier of TB and cancer
            n = 0
            while n < len(labels):
                _, pr = torch.max(output2[n].data[1:], 0)
                if pr == 0:
                    predictions2_total_changed.append(1)
                if pr == 1:
                    predictions2_total_changed.append(2)
                n += 1

            t = 0
            while t < len(labels):
                if labels[t] == 0:
                    output2[t] = torch.cuda.FloatTensor([1.0, 0.0, 0.0])
                else:
                    numberAbnormal += 1

                    _, pr = torch.max(output2[t].data[1:], 0)
                    predictions2_total.append(pr)

                    if labels[t] == 1:
                        labels2_total.append(0)
                    if labels[t] == 2:
                        labels2_total.append(1)
                t += 1

            loss2 = loss_fn2(output2, labels)
            train_loss2 += loss2.item() * images.size(0)

            loss = (1*loss1 + 1*loss2)/2
            loss.backward()
            optimizer.step()

    labels2_total = torch.Tensor(labels2_total)
    predictions2_total = torch.Tensor(predictions2_total)
    train_correct2 = (predictions2_total == labels2_total).sum().item()

    for i, p in enumerate(predictions1_total):
        if p == 0:
            predictions_total.append(0)
        if p == 1:
            pred = predictions2_total_changed[i]
            predictions_total.append(pred)
    labels_total = torch.Tensor(labels_total)
    predictions_total = torch.Tensor(predictions_total)
    train_correct = (predictions_total == labels_total).sum().item()

    # Build confusion matrix
    labels1_total = torch.tensor(labels1_total, device='cpu')
    predictions1_total = torch.tensor(predictions1_total, device='cpu')
    cf_matrix1 = confusion_matrix(labels1_total, predictions1_total)
    df_cm1 = pd.DataFrame(cf_matrix1, index=[i for i in classes1],
                          columns=[i for i in classes1])
    print(df_cm1)

    labels2_total = labels2_total.to(device='cpu')
    predictions2_total = predictions2_total.to(device='cpu')
    cf_matrix2 = confusion_matrix(labels2_total, predictions2_total)
    df_cm2 = pd.DataFrame(cf_matrix2, index=[i for i in classes2],
                          columns=[i for i in classes2])
    print(df_cm2)

    return train_loss1, train_correct1, train_loss2, train_correct2, numberAbnormal, train_correct

def valid_epoch(model, de, dataloader, loss_fn1, loss_fn2):
    valid_loss1, valid_correct1 = 0.0, 0
    valid_loss2, valid_correct2 = 0.0, 0
    valid_correct = 0
    numberAbnormal = 0

    predictions2_total_changed = []
    predictions_total = []
    labels_total = []
    predictions1_total = []
    labels1_total = []
    predictions2_total = []
    labels2_total = []

    model.eval()

    for images, labels in dataloader:
        images = images.to(de)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(de)

        for l in labels:
            labels_total.append(l)

        with torch.set_grad_enabled(False):
            output1, output2 = model(images)

            # Classifier of normal and abnormal
            labels1 = []
            c = 0
            while c < len(labels):
                if labels[c] == 2 or labels[c] == 1:
                    labels1.append(1)
                else:
                    labels1.append(0)
                c += 1

            labels1 = torch.Tensor(labels1)
            labels1 = labels1.type(torch.LongTensor)
            labels1 = labels1.to(de)

            loss1 = loss_fn1(output1, labels1)
            scores1, predictions1 = torch.max(output1.data, 1)

            valid_correct1 += (predictions1 == labels1).sum().item()
            valid_loss1 += loss1.item() * images.size(0)

            for lab in labels1:
                labels1_total.append(lab)
            for pr in predictions1:
                predictions1_total.append(pr)

            # Classifier of TB and cancer
            n = 0
            while n < len(labels):
                _, pr = torch.max(output2[n].data[1:], 0)
                if pr == 0:
                    predictions2_total_changed.append(1)
                if pr == 1:
                    predictions2_total_changed.append(2)
                n += 1

            t = 0
            while t < len(labels):
                if labels[t] == 0:
                    output2[t] = torch.cuda.FloatTensor([1.0, 0.0, 0.0])
                else:
                    numberAbnormal += 1

                    _, pr = torch.max(output2[t].data[1:], 0)
                    predictions2_total.append(pr)

                    if labels[t] == 1:
                        labels2_total.append(0)
                    if labels[t] == 2:
                        labels2_total.append(1)
                t += 1

            loss2 = loss_fn2(output2, labels)
            valid_loss2 += loss2.item() * images.size(0)

    labels2_total = torch.Tensor(labels2_total)
    predictions2_total = torch.Tensor(predictions2_total)
    valid_correct2 = (predictions2_total == labels2_total).sum().item()

    for i, p in enumerate(predictions1_total):
        if p == 0:
            predictions_total.append(0)
        if p == 1:
            pred = predictions2_total_changed[i]
            predictions_total.append(pred)

    labels_total = torch.Tensor(labels_total)
    predictions_total = torch.Tensor(predictions_total)
    valid_correct = (predictions_total == labels_total).sum().item()

    # Build confusion matrix
    labels1_total = torch.tensor(labels1_total, device='cpu')
    predictions1_total = torch.tensor(predictions1_total, device='cpu')
    cf_matrix1 = confusion_matrix(labels1_total, predictions1_total)
    df_cm1 = pd.DataFrame(cf_matrix1, index=[i for i in classes1],
                          columns=[i for i in classes1])
    print(df_cm1)

    labels2_total = labels2_total.to(device='cpu')
    predictions2_total = predictions2_total.to(device='cpu')
    cf_matrix2 = confusion_matrix(labels2_total, predictions2_total)
    df_cm2 = pd.DataFrame(cf_matrix2, index=[i for i in classes2],
                          columns=[i for i in classes2])
    print(df_cm2)

    return valid_loss1, valid_correct1, valid_loss2, valid_correct2, numberAbnormal, valid_correct

def test_epoch(model, de, dataloader, loss_fn1, loss_fn2):
    test_loss1, test_correct1 = 0.0, 0
    test_loss2, test_correct2 = 0.0, 0
    test_correct = 0
    numberAbnormal = 0

    predictions2_total_changed = []
    predictions_total = []
    labels_total = []
    predictions1_total = []
    labels1_total = []
    predictions2_total = []
    labels2_total = []

    model.eval()

    for images, labels in dataloader:
        images = images.to(de)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(de)

        for l in labels:
            labels_total.append(l)

        with torch.set_grad_enabled(False):
            output1, output2 = model(images)

            # Classifier of normal and abnormal
            labels1 = []
            c = 0
            while c < len(labels):
                if labels[c] == 2 or labels[c] == 1:
                    labels1.append(1)
                else:
                    labels1.append(0)
                c += 1

            labels1 = torch.Tensor(labels1)
            labels1 = labels1.type(torch.LongTensor)
            labels1 = labels1.to(de)

            loss1 = loss_fn1(output1, labels1)
            scores1, predictions1 = torch.max(output1.data, 1)

            test_correct1 += (predictions1 == labels1).sum().item()
            test_loss1 += loss1.item() * images.size(0)

            for lab in labels1:
                labels1_total.append(lab)
            for pr in predictions1:
                predictions1_total.append(pr)

            # Classifier of TB and cancer
            n = 0
            while n < len(labels):
                _, pr = torch.max(output2[n].data[1:], 0)
                if pr == 0:
                    predictions2_total_changed.append(1)
                if pr == 1:
                    predictions2_total_changed.append(2)
                n += 1

            t = 0
            while t < len(labels):
                if labels[t] == 0:
                    output2[t] = torch.cuda.FloatTensor([1.0, 0.0, 0.0])
                else:
                    numberAbnormal += 1

                    _, pr = torch.max(output2[t].data[1:], 0)
                    predictions2_total.append(pr)

                    if labels[t] == 1:
                        labels2_total.append(0)
                    if labels[t] == 2:
                        labels2_total.append(1)
                t += 1

            loss2 = loss_fn2(output2, labels)
            test_loss2 += loss2.item() * images.size(0)

    labels2_total = torch.Tensor(labels2_total)
    predictions2_total = torch.Tensor(predictions2_total)
    test_correct2 = (predictions2_total == labels2_total).sum().item()

    for i, p in enumerate(predictions1_total):
        if p == 0:
            predictions_total.append(0)
        if p == 1:
            pred = predictions2_total_changed[i]
            predictions_total.append(pred)

    labels_total = torch.Tensor(labels_total)
    predictions_total = torch.Tensor(predictions_total)
    test_correct = (predictions_total == labels_total).sum().item()

    # Build confusion matrix
    labels1_total = torch.tensor(labels1_total, device='cpu')
    predictions1_total = torch.tensor(predictions1_total, device='cpu')
    cf_matrix1 = confusion_matrix(labels1_total, predictions1_total)
    df_cm1 = pd.DataFrame(cf_matrix1, index=[i for i in classes1],
                          columns=[i for i in classes1])
    print(df_cm1)

    labels2_total = labels2_total.to(device='cpu')
    predictions2_total = predictions2_total.to(device='cpu')
    cf_matrix2 = confusion_matrix(labels2_total, predictions2_total)
    df_cm2 = pd.DataFrame(cf_matrix2, index=[i for i in classes2],
                          columns=[i for i in classes2])
    print(df_cm2)

    return test_loss1, test_correct1, test_loss2, test_correct2, labels_total, predictions_total, labels1_total, predictions1_total, labels2_total, predictions2_total, numberAbnormal

image_size = 224
Bronchoscopy_images = torchvision.datasets.ImageFolder(root='R:/bronchoscopy-deep learning/Bronchoscopy',
                                                   transform=torchvision.transforms.Compose([
                                                       torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Resize([image_size, image_size]),
                                                       torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                   ])
                                                   )

MainData = torch.Tensor(len(Bronchoscopy_images), 3, image_size, image_size)
MainTargets = []
for i, (img, target) in enumerate(Bronchoscopy_images):
    MainTargets.append(target)
    MainData[i, :, :, :] = img
MainTargets = torch.Tensor(MainTargets)

model = OnePartSeDensenet.densenet121(pretrained=True)

for param in model.parameters():
    param.requires_grad = True

num_ftrs1 = model.classifier1.in_features
model.classifier1 = nn.Linear(num_ftrs1, 2)
num_ftrs2 = model.classifier2.in_features
model.classifier2 = nn.Linear(num_ftrs2, 3)

model = model.to(d)

# Place of weights and loss function
skf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)

test_loss_AVG, test_acc_AVG= 0.0, 0.0
train_loss1, train_correct1 = 0.0, 0
valid_loss1, valid_correct1 = 0.0, 0
test_loss1, test_correct1 = 0.0, 0
train_loss2, train_correct2 = 0.0, 0
valid_loss2, valid_correct2 = 0.0, 0
test_loss2, test_correct2 = 0.0, 0

first_weights = copy.deepcopy(model.state_dict())
# del(model1)

total_test_labels = torch.empty(0, dtype=torch.long)  # Empty tensor for labels
total_test_preds = torch.empty(0, dtype=torch.long)   # Empty tensor for predictions
total_test_labels1 = torch.empty(0, dtype=torch.long)
total_test_preds1 = torch.empty(0, dtype=torch.long)
total_test_labels2 = torch.empty(0, dtype=torch.long)
total_test_preds2 = torch.empty(0, dtype=torch.long)


epochs_of_each_fold = []
since_train = time.time()
time_test = 0

Total_Output1 = torch.Tensor(0, 2)
Total_Output2 = torch.Tensor(0, 3)

for fold, (train_idx, test_idx) in enumerate(skf.split(Bronchoscopy_images.imgs, Bronchoscopy_images.targets)):

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    best_model_wts = copy.deepcopy(first_weights)
    model.load_state_dict(best_model_wts)

    print()
    print('Fold {}'.format(fold + 1))
    print('_.' * 20)

    val_acc = []
    val_acc_1 = []
    val_acc_2 = []
    tr_acc = []
    tr_acc_1 = []
    tr_acc_2 = []

    Data1 = torch.Tensor(len(train_idx), 3, image_size, image_size)
    targets1 = torch.Tensor(len(train_idx))
    for (i, image) in enumerate(train_idx):
        targets1[i] = MainTargets[image]
        Data1[i, :, :, :] = MainData[image, :, :, :]
    train_dataset = TensorDataset(Data1, targets1)

    train_idx, valid_idx = train_test_split(train_dataset, test_size=valid_size, random_state=42, stratify=targets1)
    Train_size = len(train_idx)

    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_idx, batch_size=train_b_s, shuffle=False,
                                               generator=torch.Generator().manual_seed(0), num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_idx, batch_size=valid_b_s,
                                               generator=torch.Generator().manual_seed(0), num_workers=0)
    test_loader = torch.utils.data.DataLoader(Bronchoscopy_images, batch_size=test_b_s, sampler=test_sampler,
                                               generator=torch.Generator().manual_seed(0), num_workers=0)

    best_acc = 0.0
    best_acc1 = 0.0
    best_acc2 = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        criterion1 = nn.CrossEntropyLoss()

        nSamples2 = [0, 350 + epoch, 1]
        normedWeights2 = [x / sum(nSamples2) for x in nSamples2]
        normedWeights2 = torch.FloatTensor(normedWeights2).to(d)
        criterion2 = nn.CrossEntropyLoss(weight=normedWeights2)

        print('confusion matrix of train data in epoch {}:'.format(epoch + 1))
        train_loss1, train_correct1, train_loss2, train_correct2, nTrainAbnormal, train_correct = train_epoch(model, d, train_loader, criterion1, criterion2, optimizer)
        print('confusion matrix of valid data in epoch {}:'.format(epoch + 1))
        valid_loss1, valid_correct1, valid_loss2, valid_correct2, nValidAbnormal, valid_correct = valid_epoch(model, d, valid_loader, criterion1, criterion2)
        print()

        train_acc = (((train_correct1 / len(train_idx)) + (train_correct2 / nTrainAbnormal)) / 2) * 100
        tr_acc.append(train_acc)
        valid_acc = (((valid_correct1/len(valid_idx))+(valid_correct2 / nValidAbnormal))/2)*100
        val_acc.append(valid_acc)

        if valid_acc > best_acc:
            best_epoch = epoch
            best_acc = valid_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print("in model Normal/AbNormal Epoch:{}/{} in Fold{}: Training Loss:{:.3f}, Valid Loss:{:.3f}   Training Acc {:.2f} %, Valid Acc {:.2f} % ".format(
            epoch + 1,
            num_epochs,
            fold+1,
            train_loss1/Train_size,
            valid_loss1/valid_size,
            (train_correct1/Train_size)*100,
            (valid_correct1/valid_size)*100))
        print('==' * 20)

        print(
            "in model TB/Cancer Epoch:{}/{} in Fold{}: Training Loss:{:.3f}, Valid Loss:{:.3f}   Training Acc {:.2f} %, Valid Acc {:.2f} % ".format(
                epoch + 1,
                num_epochs,
                fold + 1,
                train_loss2 / nTrainAbnormal,
                valid_loss2 / nValidAbnormal,
                (train_correct2 / nTrainAbnormal) * 100,
                (valid_correct2 / nValidAbnormal) * 100))
        print('==' * 20)

    epochs_of_each_fold.append(best_epoch+1)

    print(
        'best accuracy of valid dataset in fold {} is {:.2f} % in epoch: {}'.format(fold + 1, best_acc, best_epoch + 1))
    print('*_' * 10)

    # Plot
    plt.figure(figsize=(10, 8))  # Increase figure size for better resolution
    plt.plot(tr_acc, label='Training Accuracy', color='red', linewidth=2)
    plt.plot(val_acc, color='blue', label='Validation Accuracy' ,linewidth=2)

    # Increase text size
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('Accuracy', fontsize=24)
    # plt.title('ROC Curve', fontsize=16)
    plt.legend(fontsize=16, loc='best')

    # Increase font size for tick labels
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=18)

    # Save plot with high resolution
    plt.savefig(f'R:/article1/fold{fold+1}.png', dpi=100)  # Save with 300 dpi for higher resolution
    plt.show()

    model.load_state_dict(best_model_wts)

    since_test = time.time()
    Test = []
    test_loss1, test_correct1, test_loss2, test_correct2, labels_total, predictions_total, labels1_total, predictions1_total, labels2_total, predictions2_total, nTest = test_epoch(model, d, test_loader, criterion1, criterion2)
    total_test_labels = torch.cat((total_test_labels,labels_total))
    total_test_preds = torch.cat((total_test_preds,predictions_total))
    total_test_labels1 = torch.cat((total_test_labels1,labels1_total))
    total_test_preds1 = torch.cat((total_test_preds1,predictions1_total))
    total_test_labels2 = torch.cat((total_test_labels2,labels2_total))
    total_test_preds2 = torch.cat((total_test_preds2,predictions2_total))

    print('in Fold{}: Test correct1: {}, Test correct2: {}'.format(fold+1, test_correct1, test_correct2))
    print('in Fold{}: Test acc1:{:.3f} %, Test loss1 {:.2f}, Test acc2:{:.3f} %, Test loss2 {:.2f} '.format(fold + 1, (test_correct1/len(test_idx)) * 100, test_loss1 / len(test_idx), (test_correct2 / nTest) * 100, test_loss2 / nTest))

    time_test += time.time()-since_test

print('Total Test Confusion Matrix of model Normal-AbNormal :')
total_test_labels1 = torch.tensor(total_test_labels1, device='cpu')
total_test_preds1 = torch.tensor(total_test_preds1, device='cpu')
cf_matrix1 = confusion_matrix(total_test_labels1, total_test_preds1)
df_cm1 = pd.DataFrame(cf_matrix1, index=[i for i in classes1], columns=[i for i in classes1])
print(df_cm1)

print('Total Test Confusion Matrix of model TB-Cancer :')
total_test_labels2 = torch.tensor(total_test_labels2, device='cpu')
total_test_preds2 = torch.tensor(total_test_preds2, device='cpu')
cf_matrix2 = confusion_matrix(total_test_labels2, total_test_preds2)
df_cm2 = pd.DataFrame(cf_matrix2, index=[i for i in classes2], columns=[i for i in classes2])
print(df_cm2)

print('Total Test Confusion Matrix:')
total_test_labels = torch.tensor(total_test_labels, device='cpu')
total_test_preds = torch.tensor(total_test_preds, device='cpu')
cf_matrix3 = confusion_matrix(total_test_labels, total_test_preds)
df_cm3 = pd.DataFrame(cf_matrix3, index=[i for i in classes3], columns=[i for i in classes3])
print(df_cm3)


time_train = time.time() - since_train
print('Training complete in {}s and testing complete in {}s'.format(time_train, time_test))

print('epoch of each fold is ', epochs_of_each_fold)
