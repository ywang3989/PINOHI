import torch
import torch.nn as nn
import pandas as pd
# from torchdiffeq import odeint
# from TorchDiffEqPack import odesolve_adjoint_sym12
from TorchDiffEqPack.odesolver import odesolve
# import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
Material Properties & Geometry: HexPly M20 & 3M AF163-2K
    E = 65000  # Elastic modulus of adherend  [MPa]
    nu = 0.34  # Poisson ratio of adherend    [-]
    Ga = 564   # Shear modulus of adhesive    [MPa]
    tt = 2.33  # Thickness of adherend        [mm]
    L = 76.2   # Length of unnbonded region   [mm]
    b = 25.4   # Joint width                  [mm]
    c = 12.7   # Half length of bonded region [mm]
    ta = 0.05  # Thickness of adhesive        [mm]
'''


def AnalyticalStiffness(x, E, nu, Ga, ta, b, c, tt, L):
    if x <= 0:
        x = torch.tensor([0.001]).to(device)

    E = 10000*E
    Ga = 100*Ga

    D1 = E*(tt**3)*b/(12*(1-nu**2))
    D2 = 8*D1

    mu1 = torch.sqrt(x/D1)
    mu2 = torch.sqrt(x/D2)

    CC = torch.cosh(mu2*c)
    CL = torch.cosh(mu1*L)
    SC = torch.sinh(mu2*c)
    SL = torch.sinh(mu1*L)

    N = mu1*CL*SC + mu2*SL*CC - mu1*(L+c)*(mu2*CL*CC+mu1*SL*SC)

    B1 = -(tt+ta)*mu2*CC/(2*N) + (tt+ta)*(mu2*CL*CC+mu1*SL*SC)/(2*N)
    A1 = -(tt+ta)/2 - mu1*(L+c)*B1

    int_wdx = 1/4*(mu1**3)*(2*mu1*L*(A1**2-B1**2) + (A1**2+B1**2)*torch.sinh(2*mu1*L) + 2*A1*B1*(torch.cosh(2*mu1*L)-1))

    U1 = 1/(2*E)*b*((E**2)*(tt**3)*int_wdx/(12*(1-nu**2)**2) + L*tt*(x/(b*tt))**2)
    U2 = (x**2)*c/(3*E*b*tt)
    Ua = (x**2)*ta/(4*Ga*b*c)

    k = (x**2)/(2*Ua+4*(U1+U2))
    return k/1200


class MfgFunc(nn.Module):
    def __init__(self, hidden_size):
        super(MfgFunc, self).__init__()
        self.fc1 = nn.Linear(1, 4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, 16*hidden_size)
        self.fc3 = nn.Linear(16*hidden_size, 4*hidden_size)
        self.fc4 = nn.Linear(4*hidden_size, 1)

    def forward(self, x):
        out = nn.functional.elu(self.fc1(x))
        out = nn.functional.elu(self.fc2(out))
        out = nn.functional.elu(self.fc3(out))
        out = self.fc4(out)
        return out


class ODEFunc(nn.Module):
    def __init__(self, hidden_size):
        super(ODEFunc, self).__init__()
        self.mfgfunc = MfgFunc(hidden_size)
        self.alyfunc = AnalyticalStiffness

    def forward(self, t, x):
        return self.mfgfunc(x)  # + 1/12000*self.alyfunc(x)


class MLPstroke(nn.Module):
    def __init__(self, hidden_size):
        super(MLPstroke, self).__init__()
        self.fc1 = nn.Linear(4*hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = nn.functional.relu(self.fc1(x))
        out = nn.functional.relu(self.fc2(out))
        return out


options = {}
# options.update({'method': 'sym12async'})
options.update({'method': 'dopri5'})
options.update({'h': None})
options.update({'t0': 0.0})
options.update({'rtol': 1e-3})
options.update({'atol': 1e-3})
options.update({'print_neval': False})
options.update({'neval_max': 1000000})
options.update({'safety': None})
options.update({'interpolation_method': 'cubic'})
options.update({'regenerate_graph': False})


class NeuralODE(nn.Module):
    def __init__(self, hidden_size):
        super(NeuralODE, self).__init__()
        self.hidden_size = hidden_size
        self.func = ODEFunc(hidden_size)
        # self.stroke = MLPstroke(hidden_size)

        self.fc = nn.Linear(4, hidden_size)
        self.coef = nn.Linear(4*hidden_size, hidden_size)
        self.initial = nn.Linear(4*hidden_size, hidden_size)
        self.readout = nn.Linear(hidden_size, 1)

        self.conv_temp = nn.Conv2d(1, hidden_size, 2)
        self.conv_caA = nn.Conv2d(1, hidden_size, 3)
        self.conv_caB = nn.Conv2d(1, hidden_size, 3)

    def forward(self, t, x):
        temp_out = nn.functional.elu(self.conv_temp(x[0, 5:9].view((1, 1, 2, 2)))).view(-1)
        caA_out = nn.functional.elu(self.conv_caA(x[0, 9:18].view((1, 1, 3, 3)))).view(-1)
        caB_out = nn.functional.elu(self.conv_caB(x[0, 18:27].view((1, 1, 3, 3)))).view(-1)

        out = torch.cat((nn.functional.elu(self.fc(x[0, 1:5])), temp_out, caA_out, caB_out))
        # stroke_pred = self.stroke(out)

        coef = nn.functional.elu(self.coef(out))
        x0 = nn.functional.elu(self.initial(out))
        z0 = nn.functional.elu(torch.matmul(coef, x0).view(-1))

        # z = odeint(self.func, z0, t)

        options.update({'t_eval': t.cpu().tolist()})
        options.update({'t1': t.cpu().tolist()[-1]})

        z = odesolve(self.func, z0, options=options)

        # t_list = t.cpu().tolist()
        # z = torch.zeros(t.size(dim=0), 1).to(device)

        # for i in range(t.size(dim=0)-1):
        #    options.update({'t0': t[i].cpu().numpy()})
        #    options.update({'t1': t[i+1].cpu().numpy()})
        #    z[i+1, 0] = odesolve_adjoint_sym12(self.func, z[i], options=options)

        # output = nn.functional.relu(self.readout(z))

        return z  # , stroke_pred


class ResNetBlock(nn.Module):
    def __init__(self, hidden_size,):
        super(ResNetBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        return self.ln(self.net(x)) + x


def GenerateNet(layers, input_size, output_size, hidden_size):
    resnet = [nn.Linear(input_size, hidden_size), ]

    for _ in range(layers):
        resnet.append(nn.LeakyReLU())
        resnet.append(ResNetBlock(hidden_size))

    resnet.append(nn.LeakyReLU())
    resnet.append(nn.Linear(hidden_size, output_size))

    return nn.Sequential(*resnet)


class PINOHI(nn.Module):
    def __init__(self, layers, input_size, output_size, hidden_size, feature_size):
        super(PINOHI, self).__init__()
        self.initial = nn.Linear(1, feature_size)  # For NODE+Mfg & NODE+Mfg+Ana
        self.readout = nn.Linear(output_size, 1)
        self.scaling = nn.Linear(1, 1)

        self.fab_fc = nn.Linear(4, feature_size)
        self.dim_fc = nn.Linear(5, feature_size)
        self.mfg_fc = nn.Linear(7, feature_size)
        self.conv_tempF = nn.Conv2d(1, feature_size, 2)
        self.conv_tempJ = nn.Conv2d(1, feature_size, 2)
        self.conv_caA = nn.Conv2d(1, feature_size, 2)
        self.conv_caB = nn.Conv2d(1, feature_size, 2)

        self.youngsC = nn.Linear(2*feature_size, 1)
        self.poissonC = nn.Linear(2*feature_size, 1)
        self.shearA = nn.Linear(4*feature_size, 1)

        self.analyfunc = AnalyticalStiffness
        self.resnet = GenerateNet(layers, input_size, output_size, hidden_size)

    def forward(self, x):
        '''
        x: 39-dim vector, where
            0:    - scaled load: F
            1~4   - norm fab para: t_outC, p_F, Te_F, He_F
            5~8   - norm fab temp: T_F
            9~13  - norm geo dim:  ta, b, c, tt, L
            14~18 - orig geo dim:  ta, b, c, tt, L
            19~25 - norm joi para: t_outA, r_J, t_J, p_J, Te_J, He_J, FLASH
            26~29 - norm joi temp: T_J
            30~33 - norm cont ang: PsiA
            34~37 - norm cont ang: PsiB
            38    - placeholder: 0
        '''

        # Original geometric dimension in Ana & NODE+Mfg+Ana
        ta = x[0, 14]
        b = x[0, 15]
        c = x[0, 16]
        tt = x[0, 17]
        L = x[0, 18]

        h = nn.functional.leaky_relu(self.initial(x[0, 0].view(-1)))
        z_dim = nn.functional.leaky_relu(self.dim_fc(x[0, 9:14]))
        z_xi = nn.functional.leaky_relu(self.mfg_fc(x[0, 19:26]))
        z_tempJ = nn.functional.leaky_relu(self.conv_tempJ(x[0, 26:30].view((1, 1, 2, 2)))).view(-1)
        z_caA = nn.functional.leaky_relu(self.conv_caA(x[0, 30:34].view((1, 1, 2, 2)))).view(-1)
        z_caB = nn.functional.leaky_relu(self.conv_caB(x[0, 34:38].view((1, 1, 2, 2)))).view(-1)

        z_mfg = torch.vstack((z_dim, z_xi, z_tempJ, z_caA, z_caB))
        g = torch.matmul(z_mfg, h)
        dhdt = self.resnet(g)  # For NODE+Mfg & NODE+Mfg+Ana

        z_fab = nn.functional.leaky_relu(self.fab_fc(x[0, 1:5]))
        z_tempF = nn.functional.leaky_relu(self.conv_tempF(x[0, 5:9].view((1, 1, 2, 2)))).view(-1)
        ztilde_fab = torch.cat((z_fab, z_tempF))
        ztilde_mfg = torch.cat((z_xi, z_tempJ, z_caA, z_caB))
        E = torch.clamp(nn.functional.softplus(self.youngsC(ztilde_fab)), min=3.25, max=9.75)  # (1±0.5)*6.5
        nu = torch.clamp(nn.functional.softplus(self.poissonC(ztilde_fab)), min=0.17, max=0.51)  # (1±0.5)*0.34
        Ga = torch.clamp(nn.functional.softplus(self.shearA(ztilde_mfg)), min=2.82, max=8.46)  # (1±0.5)*5.64

        dFdt_NN = self.readout(dhdt)
        dFdt_Ana = self.scaling(self.analyfunc(x[0, 0].view(-1), E, nu, Ga, ta, b, c, tt, L))
        dFdt = dFdt_NN + dFdt_Ana

        return dFdt


class PINOHIwoAna(nn.Module):
    def __init__(self, layers, input_size, output_size, hidden_size, feature_size):
        super(PINOHIwoAna, self).__init__()
        self.initial = nn.Linear(1, feature_size)  # For NODE+Mfg & NODE+Mfg+Ana
        self.readout = nn.Linear(output_size, 1)
        self.dim_fc = nn.Linear(5, feature_size)
        self.mfg_fc = nn.Linear(7, feature_size)
        self.conv_tempJ = nn.Conv2d(1, feature_size, 2)
        self.conv_caA = nn.Conv2d(1, feature_size, 2)
        self.conv_caB = nn.Conv2d(1, feature_size, 2)

        self.resnet = GenerateNet(layers, input_size, output_size, hidden_size)

    def forward(self, x):
        '''
        x: 39-dim vector, where
            0:    - scaled load: F
            1~4   - norm fab para: t_outC, p_F, Te_F, He_F
            5~8   - norm fab temp: T_F
            9~13  - norm geo dim:  ta, b, c, tt, L
            14~18 - orig geo dim:  ta, b, c, tt, L
            19~25 - norm joi para: t_outA, r_J, t_J, p_J, Te_J, He_J, FLASH
            26~29 - norm joi temp: T_J
            30~33 - norm cont ang: PsiA
            34~37 - norm cont ang: PsiB
            38    - placeholder: 0
        '''

        h = nn.functional.leaky_relu(self.initial(x[0, 0].view(-1)))
        z_dim = nn.functional.leaky_relu(self.dim_fc(x[0, 9:14]))
        z_xi = nn.functional.leaky_relu(self.mfg_fc(x[0, 19:26]))
        z_tempJ = nn.functional.leaky_relu(self.conv_tempJ(x[0, 26:30].view((1, 1, 2, 2)))).view(-1)
        z_caA = nn.functional.leaky_relu(self.conv_caA(x[0, 30:34].view((1, 1, 2, 2)))).view(-1)
        z_caB = nn.functional.leaky_relu(self.conv_caB(x[0, 34:38].view((1, 1, 2, 2)))).view(-1)
        z_mfg = torch.vstack((z_dim, z_xi, z_tempJ, z_caA, z_caB))
        g = torch.matmul(z_mfg, h)
        dhdt = self.resnet(g)  # For NODE+Mfg & NODE+Mfg+Ana

        dFdt_NN = self.readout(dhdt)

        return dFdt_NN


class CalibAna(nn.Module):
    def __init__(self, layers, input_size, output_size, hidden_size, feature_size):
        super(CalibAna, self).__init__()
        self.scaling = nn.Linear(1, 1)
        self.youngsC = nn.Linear(8, 1)
        self.poissonC = nn.Linear(8, 1)
        self.shearA = nn.Linear(19, 1)

        self.analyfunc = AnalyticalStiffness

    def forward(self, x):
        '''
        x: 39-dim vector, where
            0:    - scaled load: F
            1~4   - norm fab para: t_outC, p_F, Te_F, He_F
            5~8   - norm fab temp: T_F
            9~13  - norm geo dim:  ta, b, c, tt, L
            14~18 - orig geo dim:  ta, b, c, tt, L
            19~25 - norm joi para: t_outA, r_J, t_J, p_J, Te_J, He_J, FLASH
            26~29 - norm joi temp: T_J
            30~33 - norm cont ang: PsiA
            34~37 - norm cont ang: PsiB
            38    - placeholder: 0
        '''

        # Original geometric dimension in Ana & NODE+Mfg+Ana
        ta = x[0, 14]
        b = x[0, 15]
        c = x[0, 16]
        tt = x[0, 17]
        L = x[0, 18]
        E = torch.clamp(nn.functional.softplus(self.youngsC(x[0, 1:9])), min=3.25, max=9.75)
        nu = torch.clamp(nn.functional.softplus(self.poissonC(x[0, 1:9])), min=0.17, max=0.51)
        Ga = torch.clamp(nn.functional.softplus(self.shearA(x[0, 19:38])), min=2.82, max=8.46)

        dFdt_Ana = self.scaling(self.analyfunc(x[0, 0].view(-1), E, nu, Ga, ta, b, c, tt, L))

        return dFdt_Ana


class NODE(nn.Module):
    def __init__(self, layers, input_size, output_size, hidden_size, feature_size):
        super(NODE, self).__init__()
        self.initial = nn.Linear(1, input_size)  # For NODE
        self.readout = nn.Linear(output_size, 1)

        self.resnet = GenerateNet(layers, input_size, output_size, hidden_size)

    def forward(self, x):
        '''
        x: 39-dim vector, where
            0:    - scaled load: F
            1~4   - norm fab para: t_outC, p_F, Te_F, He_F
            5~8   - norm fab temp: T_F
            9~13  - norm geo dim:  ta, b, c, tt, L
            14~18 - orig geo dim:  ta, b, c, tt, L
            19~25 - norm joi para: t_outA, r_J, t_J, p_J, Te_J, He_J, FLASH
            26~29 - norm joi temp: T_J
            30~33 - norm cont ang: PsiA
            34~37 - norm cont ang: PsiB
            38    - placeholder: 0
        '''

        h = nn.functional.leaky_relu(self.initial(x[0, 0].view(-1)))
        dhdt = self.resnet(h)  # For NODE

        dFdt_NN = self.readout(dhdt)

        return dFdt_NN


def LoadData(path, step=1):
    df = pd.read_csv(path, header=None)

    F = df.values[0::step, 0]/1200
    F = F.astype('float32')
    F = torch.from_numpy(F.reshape((len(F), 1)))

    t = df.values[0::step, 1]
    t = t.astype('float32')
    t = torch.from_numpy(t)

    mfg = df.values[0, 2:-1]
    mfg = mfg.astype('float32')
    mfg = torch.from_numpy(mfg)

    return F, t, mfg


def TrainIndexSelection(training_size, testing_batch):
    ''' # Batch #7 - First round of SA
    if testing_batch == 7:
        if training_size == 60:  # 60: P33, S3, C24
            train_ind = [6, 7, 9, 10, 11, 13, 15, 16, 18, 19, 20, 26, 27, 28, 30, 33, 34, 35, 39, 40, 47,
                         48, 49, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 69, 70, 71,
                         72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 88, 89, 90, 92, 93]
        elif training_size == 50:  # 50: P28, S3, C19
            train_ind = [6, 7, 10, 11, 15, 16, 18, 19, 20, 26, 27, 30, 33, 34, 39, 40, 47, 48, 49, 51, 52,
                         53, 55, 56, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 72, 73, 75, 76, 77, 80,
                         81, 82, 83, 85, 86, 88, 89, 90, 92]
        elif training_size == 40:  # 40: P23, S3, C14
            train_ind = [6, 10, 11, 15, 16, 18, 19, 20, 27, 30, 33, 40, 47, 48, 49, 52, 53, 55, 56, 58, 59,
                         61, 62, 65, 66, 69, 70, 71, 73, 75, 76, 77, 80, 81, 82, 83, 85, 88, 90, 92]
        elif training_size == 30:  # 30: P16, S3, C11
            train_ind = [6, 10, 11, 16, 18, 19, 20, 27, 30, 33, 40, 48, 49, 53, 56, 59, 61, 62, 66, 69, 70,
                         71, 75, 76, 77, 80, 83, 85, 90, 92]
        elif training_size == 20:  # 20: P8, S2, C10
            train_ind = [6, 10, 18, 20, 30, 33, 40, 48, 49, 56, 59, 62, 69, 70, 75, 76, 80, 83, 85, 92]
        elif training_size == 10:  # 10: P4, S1, C5
            train_ind = [10, 18, 30, 40, 48, 56, 59, 75, 76, 85]
        else:
            train_ind = []
            print('Please enter training_size from one of 10, 20, 30, 40, 50, 60')
    '''

    ''' # Batch #7 - Second round of SA
    if testing_batch == 7:
        if training_size == 60:  # 60: P33, S3, C24
            train_ind = [2, 5, 6, 7, 9, 10, 12, 13, 15, 16, 19, 20, 25, 26, 29, 30, 34, 35, 39, 40, 46, 47,
                         48, 49, 51, 53, 55, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
                         74, 75, 76, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93]
        elif training_size == 50:  # 50: P28, S3, C19
            train_ind = [2, 5, 6, 7, 10, 11, 13, 16, 19, 20, 25, 26, 27, 28, 34, 35, 36, 48, 49, 50, 51, 52,
                         55, 56, 57, 58, 60, 61, 62, 64, 65, 66, 69, 70, 71, 72, 73, 76, 77, 79, 80, 81, 82,
                         83, 85, 86, 87, 89, 90, 93]
        elif training_size == 40:  # 40: P23, S3, C14
            train_ind = [2, 5, 7, 11, 12, 13, 16, 20, 26, 28, 33, 34, 35, 36, 40, 46, 47, 48, 52, 55, 56, 57,
                         58, 61, 62, 65, 66, 67, 69, 71, 72, 74, 76, 80, 81, 82, 83, 87, 89, 90]
        elif training_size == 30:  # 30: P16, S3, C11
            train_ind = [2, 5, 6, 7, 9, 13, 18, 19, 27, 28, 30, 33, 46, 49, 50, 51, 52, 53, 56, 58, 59, 61, 65,
                         66, 71, 74, 79, 80, 83, 86]
        elif training_size == 20:  # 20: P8, S2, C10
            train_ind = [9, 11, 18, 19, 20, 28, 30, 34, 35, 51, 52, 55, 57, 68, 69, 70, 77, 86, 91, 93]
        elif training_size == 10:  # 10: P4, S1, C5
            train_ind = [12, 20, 30, 33, 34, 51, 75, 87, 88, 91]
        else:
            train_ind = []
            print('Please enter training_size from one of 10, 20, 30, 40, 50, 60')
    '''

    # Batch #7 - Third round of SA
    if testing_batch == 7:
        if training_size == 60:  # 60: P33, S3, C24
            train_ind = [2, 5, 7, 9, 10, 11, 13, 15, 16, 18, 19, 26, 30, 33, 34, 35, 36, 39, 40, 46, 47, 48,
                         49, 50, 51, 52, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72,
                         73, 74, 75, 76, 77, 79, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93]
        elif training_size == 50:  # 50: P28, S3, C19
            train_ind = [2, 6, 9, 10, 12, 13, 15, 18, 19, 20, 25, 26, 28, 29, 33, 35, 36, 39, 40, 46, 47, 48,
                         51, 53, 56, 58, 59, 60, 61, 62, 63, 65, 67, 68, 70, 71, 72, 74, 77, 79, 82, 83, 84,
                         85, 86, 87, 88, 90, 91, 93]
        elif training_size == 40:  # 40: P23, S3, C14
            train_ind = [2, 5, 6, 10, 11, 13, 15, 16, 19, 27, 29, 34, 35, 36, 39, 40, 46, 47, 48, 51, 52, 53,
                         56, 58, 59, 60, 63, 67, 69, 70, 71, 72, 73, 81, 83, 84, 85, 86, 87, 90]
        elif training_size == 30:  # 30: P16, S3, C11
            train_ind = [2, 6, 9, 10, 11, 12, 19, 25, 30, 34, 35, 46, 50, 57, 59, 62, 63, 64, 65, 67, 68, 72,
                         74, 82, 83, 84, 89, 91, 92, 93]
        elif training_size == 20:  # 20: P8, S2, C10
            train_ind = [6, 12, 16, 28, 34, 40, 46, 55, 56, 59, 61, 67, 69, 70, 71, 75, 80, 82, 89, 93]
        elif training_size == 10:  # 10: P4, S1, C5
            train_ind = [7, 11, 13, 27, 28, 39, 63, 79, 85, 89]
        else:
            train_ind = []
            print('Please enter training_size from one of 10, 20, 30, 40, 50, 60')

    '''
    elif testing_batch == 9:
        if training_size == 60:  # 60: P33, S3, C24
            train_ind = [6, 7, 9, 10, 11, 13, 15, 16, 18, 19, 20, 26, 27, 28, 30, 32, 33, 34, 35, 39, 40,
                         41, 42, 44, 45, 47, 48, 49, 51, 58, 59, 60, 61, 62, 63, 65, 66, 67, 69, 70, 71,
                         72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 88, 89, 90, 92, 93]
        elif training_size == 50:  # 50: P28, S3, C19
            train_ind = [6, 7, 10, 11, 15, 16, 18, 19, 20, 26, 27, 30, 32, 33, 34, 39, 40, 41, 44, 45, 47,
                         48, 49, 51, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 72, 73, 75, 76, 77, 80,
                         81, 82, 83, 85, 86, 88, 89, 90, 92]
        elif training_size == 40:  # 40: P23, S3, C14
            train_ind = [6, 10, 11, 15, 16, 18, 19, 20, 27, 30, 32, 33, 40, 41, 44, 45, 47, 48, 49, 58, 59,
                         61, 62, 65, 66, 69, 70, 71, 73, 75, 76, 77, 80, 81, 82, 83, 85, 88, 90, 92]
        elif training_size == 30:  # 30: P16, S3, C11
            train_ind = [6, 10, 11, 16, 18, 19, 20, 27, 30, 32, 33, 40, 44, 48, 49, 59, 61, 62, 66, 69, 70,
                         71, 75, 76, 77, 80, 83, 85, 90, 92]
        elif training_size == 20:  # 20: P8, S2, C10
            train_ind = [6, 10, 18, 20, 30, 33, 40, 44, 48, 49, 59, 62, 69, 70, 75, 76, 80, 83, 85, 92]
        elif training_size == 10:  # 10: P4, S1, C5
            train_ind = [10, 18, 30, 40, 44, 48, 59, 75, 76, 85]
        else:
            train_ind = []
            print('Please enter training_size from one of 10, 20, 30, 40, 50, 60')
    elif testing_batch == 13:
        if training_size == 60:  # 60: P33, S3, C24
            train_ind = [6, 7, 9, 10, 11, 13, 15, 16, 18, 19, 20, 26, 27, 28, 30, 32, 33, 34, 35, 39, 40,
                         41, 42, 44, 45, 47, 48, 49, 51, 52, 53, 55, 56, 57,  58, 59, 60, 61, 62, 63, 65,
                         66, 67, 69, 70, 71, 72, 73, 74, 75, 82, 83, 84, 85, 86, 88, 89, 90, 92, 93]
        elif training_size == 50:  # 50: P28, S3, C19
            train_ind = [6, 7, 10, 11, 15, 16, 18, 19, 20, 26, 27, 30, 32, 33, 34, 39, 40, 41, 44, 45, 47,
                         48, 49, 51, 52, 53, 56, 57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 72, 73,
                         75, 82, 83, 85, 86, 88, 89, 90, 92]
        elif training_size == 40:  # 40: P23, S3, C14
            train_ind = [6, 10, 11, 15, 16, 18, 19, 20, 27, 30, 32, 33, 40, 41, 44, 45, 47, 48, 49, 52, 53,
                         56, 57, 58, 59, 61, 62, 65, 66, 69, 70, 71, 73, 75, 82, 83, 85, 88, 90, 92]
        elif training_size == 30:  # 30: P16, S3, C11
            train_ind = [6, 10, 11, 16, 18, 19, 20, 27, 30, 32, 33, 40, 44, 48, 49, 52, 53, 57, 59, 61, 62,
                         66, 69, 70, 71, 75, 83, 85, 90, 92]
        elif training_size == 20:  # 20: P8, S2, C10
            train_ind = [6, 10, 18, 20, 30, 33, 40, 44, 48, 49, 53, 57, 59, 62, 69, 70, 75, 83, 85, 92]
        elif training_size == 10:  # 10: P4, S1, C5
            train_ind = [10, 18, 30, 40, 44, 48, 53, 59, 75, 85]
        else:
            train_ind = []
            print('Please enter training_size from one of 10, 20, 30, 40, 50, 60')
    else:
        print('Please enter testing_batech from one of 7, 9, 13')
    '''

    return train_ind


def LeaveOneBatchOutCrossValidation(leave_one_out_index):
    '''
    leave_one_out_index should be 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    '''
    batch_7 = [32, 41, 42, 44, 45]
    batch_8 = [46, 47, 48, 49, 50, 51]
    batch_9 = [52, 53, 55, 56, 57]
    batch_10 = [58, 59, 60, 61, 62, 63]
    batch_11 = [64, 65, 66, 67, 68, 69]
    batch_12 = [70, 71, 72, 73, 74, 75]
    batch_13 = [76, 77, 79, 80, 81]
    batch_14 = [82, 83, 84, 85, 86, 87]
    batch_15 = [88, 89, 90, 91, 95, 96]
    batches = [batch_7, batch_8, batch_9, batch_10, batch_11, batch_12, batch_13, batch_14, batch_15]

    train_ind = [2, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 39, 40]
    # train_ind = [25, 26, 27, 28, 29, 30]

    if leave_one_out_index in [7, 8, 9, 10, 11, 12, 13, 14, 15]:
        test_ind = batches[leave_one_out_index-7]
        for i in range(9):
            if i != leave_one_out_index-7:
                train_ind = train_ind + batches[i]
    elif leave_one_out_index == 6:
        test_ind = train_ind
        train_ind = batch_7 + batch_8 + batch_9 + batch_10 + batch_11 + batch_12 + batch_13 + batch_14 + batch_15
    else:
        print('Please enter one of 6, 7, 8, 9, 10, 11, 12, 13, 14, 15')

    return train_ind, test_ind


def MAPECompute(true, pred):
    mape = torch.mean(torch.abs(true-pred)/true)
    return mape.item()


def MSPECompute(true, pred):
    mspe = torch.mean(torch.pow((true-pred)/true, 2))
    return mspe.item()


def MARECompute(true, pred, timepoint):
    mare = torch.trapezoid(torch.abs(true-pred), timepoint)/torch.trapezoid(true, timepoint)
    return mare.item()


def CosineSimilarity(true, pred):
    cosine = torch.matmul(true, pred)/(torch.norm(true, p=2)*torch.norm(pred, p=2))
    return cosine.item()
