Data = readtable('Result_SA_7.xlsx');

MARE = [];
MAPE = [];

% PINOHI
for i = 0:6
    MARE = [MARE;table2array(Data(17-i,4:18))'];
    MAPE = [MAPE;table2array(Data(24-i,4:18))'];
end
% Neural ODE + Mfg
for i = 0:6
    MARE = [MARE;table2array(Data(31-i,4:18))'];
    MAPE = [MAPE;table2array(Data(38-i,4:18))'];
end

% Model = repelem({'PINOHI';'Neural ODE + Mfg.';'Calib. Ana.';'Neural ODE'},105,1);
Model = repelem({'PINOHI';'Neural ODE + Mfg.'},105,1);
Samples = repelem([1;2;3;4;5;6;7.2],15);
% Samples = [Samples;Samples;Samples;Samples];
Samples = [Samples;Samples];
TestingData = table(Model,Samples,MARE,MAPE);


% Testing mMARE
figure;
boxchart(TestingData.Samples,TestingData.MARE,'GroupByColor',TestingData.Model);
axis([0.4,7.8,0.05,0.3]);
ylabel('Batch #7 Testing mMARE');
xlabel('Number of Samples in Training (x10)');
legend;
box on;
