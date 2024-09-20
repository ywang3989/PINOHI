# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from function import PINOHI, PINOHIwoAna, CalibAna, NODE, LoadData, MSPECompute, LeaveOneBatchOutCrossValidation
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# training_size = 50
# testing_batch = 7
# train_ind = TrainIndexSelection(training_size, testing_batch)
# test_ind = [32, 41, 42, 44, 45]  # Batch # 7

leave_one_out_index = 7  # {6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
train_ind, test_ind = LeaveOneBatchOutCrossValidation(leave_one_out_index)

layers = 5
input_size = 5
output_size = 5
hidden_size = 128
feature_size = 16

model_type = 'PINOHI'
weight_file_name = model_type + '.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if model_type == 'PINOHI':
    model = PINOHI(layers, input_size, output_size, hidden_size, feature_size).to(device)
elif model_type == 'PINOHI_wo_Ana':
    model = PINOHIwoAna(layers, input_size, output_size, hidden_size, feature_size).to(device)
elif model_type == 'Calib_Ana':
    model = CalibAna(layers, input_size, output_size, hidden_size, feature_size).to(device)
elif model_type == 'NODE':
    model = NODE(layers, input_size, output_size, hidden_size, feature_size).to(device)

model.load_state_dict(torch.load(weight_file_name))
# print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0000625, weight_decay=1e-4)  # 88
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20)

history_train = []
# history_test = []
step = 1
epsilon = 1e-3
Lambda = 0.5
epoches = 40
for epoch in range(epoches):
    train_Loss, test_Loss = torch.zeros(1).to(device), torch.zeros(1).to(device)

    random.shuffle(train_ind)

    model.train()
    for j in train_ind:
        path = "Training_CV/Train_DIC_geo_norm_" + str(j) + ".csv"
        F, t, mfg = LoadData(path, step=step)
        F = F.to(device)
        t = t.to(device)

        optimizer.zero_grad()

        Fhat = torch.zeros(F.size()).to(device)
        for k, F_k in enumerate(F):
            input = torch.cat((Fhat[k].cpu(), mfg)).view((1, len(mfg)+1)).to(device)
            dFdt = model(input)
            Fhat[k+1, 0] = Fhat[k, 0] + dFdt*(t[k+1] - t[k])
            if math.isnan(Fhat[k+1, 0]):
                print(dFdt)
                break

            if k == F.size(dim=0)-2:
                break

        train_loss = Lambda*criterion(Fhat, F) + (1-Lambda)*MSPECompute(F[1:, 0], Fhat[1:, 0])

        train_loss.backward()
        optimizer.step()

        train_Loss = torch.add(train_Loss, train_loss)

    '''
    model.eval()
    for j in test_ind:
        path = "Train_norm_" + str(j) + ".csv"
        F, t, mfg = LoadData(path, step=step)
        F = F.to(device)
        t = t.to(device)

        Fhat = torch.zeros(F.size()).to(device)
        for k, F_k in enumerate(F):
            input = torch.cat((Fhat[k].cpu(), mfg)).view((1, len(mfg)+1)).to(device)
            dFdt = model(input)
            Fhat[k+1, 0] = Fhat[k, 0] + dFdt*(t[k+1] - t[k])

            if k == F.size(dim=0)-2:
                break

        test_loss = Lambda*criterion(Fhat, F) + (1-Lambda)*MSPECompute(F[1:, 0], Fhat[1:, 0])
        test_Loss = torch.add(test_Loss, test_loss)
    '''

    history_train.append(train_Loss/len(train_ind))
    # history_test.append(test_Loss/len(test_ind))

    scheduler.step(train_loss/len(train_ind))

    # torch.save(model.state_dict(), weight_file_name)

    if epoch % 1 == 0:
        print('Epoch #', epoch + 1,
              '| lr: ', optimizer.param_groups[0]['lr'],
              '| Training Loss: ', history_train[epoch].item())  # 'testing: ', history_test[epoch].item())

    if epoch >= 1 and torch.abs(history_train[epoch]-history_train[epoch-1])/history_train[epoch-1] <= epsilon:
        break

torch.save(model.state_dict(), weight_file_name)

train_losses = np.array([loss.cpu().detach().numpy() for loss in history_train])
plt.plot(train_losses)
plt.show()
# np.savetxt("aTrain_loss.csv", train_losses)

# test_losses = np.array([loss.cpu().detach().numpy() for loss in history_test])
# plt.plot(test_losses)
# plt.show()
# np.savetxt("Test_loss.csv", test_losses)

'''
i = 1
model_ = PINOHI(layers, input_size, output_size, hidden_size, feature_size).to(device)
model_.load_state_dict(torch.load(weight_file_name))
model_.eval()
for j in train_ind:
    path = "Train_DIC_geo_norm_" + str(j) + ".csv"
    F, t, mfg = LoadData(path, step)
    F = F.to(device)
    t = t.to(device)

    Fhat = torch.zeros(F.size()).to(device)
    Fhat[0, 0] = F[0, 0]
    for k, F_k in enumerate(F):
        input = torch.cat((Fhat[k].cpu(), mfg)).view((1, len(mfg)+1)).to(device)
        dFdt = model_(input)
        Fhat[k+1, 0] = Fhat[k, 0] + dFdt*(t[k+1] - t[k])

        if k == F.size(dim=0)-2:
            break

    plt.subplot(4, 5, i)
    plt.plot(t.cpu().detach().numpy(), 1200 * Fhat.cpu().detach().numpy(), 'k')
    plt.plot(t.cpu().detach().numpy(), 1200 * F.cpu().detach().numpy(), 'r')

    i = i+1

plt.show()
'''
