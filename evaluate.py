from function import PINOHI, PINOHIwoAna, CalibAna, NODE, LoadData, MAPECompute, MARECompute, LeaveOneBatchOutCrossValidation
import torch
import matplotlib.pyplot as plt
import numpy as np

# training_size = 50
# testing_batch = 7
# train_ind = TrainIndexSelection(training_size, testing_batch)
# test_ind = [32, 41, 42, 44, 45]  # Batch # 7

leave_one_out_index = 7  # {6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
train_ind, test_ind = LeaveOneBatchOutCrossValidation(leave_one_out_index)

train_mare = []
test_mare = []
train_mape = []
test_mape = []
step = 1

layers = 5
input_size = 5
output_size = 5
hidden_size = 128
feature_size = 16

model_type = 'PINOHI'
weight_file_name = model_type + '.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if model_type == 'PINOHI':
    model_ = PINOHI(layers, input_size, output_size, hidden_size, feature_size).to(device)
elif model_type == 'PINOHI_wo_Ana':
    model_ = PINOHIwoAna(layers, input_size, output_size, hidden_size, feature_size).to(device)
elif model_type == 'Calib_Ana':
    model_ = CalibAna(layers, input_size, output_size, hidden_size, feature_size).to(device)
elif model_type == 'NODE':
    model_ = NODE(layers, input_size, output_size, hidden_size, feature_size).to(device)

model_.load_state_dict(torch.load(weight_file_name))
model_.eval()

i = 1
for j in train_ind:
    path = "Training_CV/Train_DIC_geo_norm_" + str(j) + ".csv"
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

    train_mare.append(MARECompute(F[0:, 0], Fhat[0:, 0], t))
    train_mape.append(MAPECompute(F[1:, 0], Fhat[1:, 0]))

    plt.subplot(8, 9, i)
    plt.ylabel('')
    plt.xlabel('')
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.plot(t.cpu().detach().numpy(), 1200 * Fhat.cpu().detach().numpy(), 'k')
    plt.plot(t.cpu().detach().numpy(), 1200 * F.cpu().detach().numpy(), 'r')

    i = i+1

plt.show()

i = 1
for j in test_ind:
    path = "Training_CV/Train_DIC_geo_norm_" + str(j) + ".csv"
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

    test_mare.append(MARECompute(F[0:, 0], Fhat[0:, 0], t))
    test_mape.append(MAPECompute(F[1:, 0], Fhat[1:, 0]))

    plt.subplot(2, 3, i)
    plt.plot(t.cpu().detach().numpy(), 1200 * Fhat.cpu().detach().numpy(), 'k')
    plt.plot(t.cpu().detach().numpy(), 1200 * F.cpu().detach().numpy(), 'r')

    i = i+1

plt.show()

'''
i = 1
for j in test_ind_2:
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

    test_mare_2.append(MARECompute(F[0:, 0], Fhat[0:, 0], t))
    test_mape_2.append(MAPECompute(F[1:, 0], Fhat[1:, 0]))

    if training_size == 10:
        plt.subplot(3, 4, i)
    else:
        plt.subplot(2, 3, i)
    # plt.subplot(2, 3, i)
    plt.plot(t.cpu().detach().numpy(), 1200 * Fhat.cpu().detach().numpy(), 'k')
    plt.plot(t.cpu().detach().numpy(), 1200 * F.cpu().detach().numpy(), 'r')

    i = i+1

plt.show()
'''

print('')
print('Train mean MARE: ', round(np.mean(train_mare), 5), '| median MARE: ', round(np.median(train_mare), 5), '| std: ', round(np.std(train_mare), 5))
print('Train mean MAPE: ', round(np.mean(train_mape), 5), '| median MAPE: ', round(np.median(train_mape), 5), '| std: ', round(np.std(train_mape), 5))
print('Test mean MARE: ', round(np.mean(test_mare), 5), '| median MARE: ', round(np.median(test_mare), 5), '| std: ', round(np.std(test_mare), 5))
print('Test mean MAPE: ', round(np.mean(test_mape), 5), '| median MAPE: ', round(np.median(test_mape), 5), '| std: ', round(np.std(test_mape), 5))
print('Test MARE: ', np.round(test_mare, 5))
print('Test MAPE: ', np.round(test_mape, 5))
print('')
