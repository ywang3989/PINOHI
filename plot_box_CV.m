%           6       7     8      9     10     11     12     13     14     15
mMARE = [0.2052,0.1131,0.1203,0.0772,0.0910,0.0724,0.0627,0.1069,0.0840,0.1091;  % pinohi
         0.2238,0.1289,0.1495,0.1151,0.0950,0.0986,0.0883,0.1654,0.1471,0.1660;  % piwoan
         0.2765,0.1565,0.1688,0.1864,0.1010,0.0844,0.0687,0.1133,0.0910,0.1594;  % calian
         0.2517,0.1269,0.1349,0.1581,0.0955,0.1056,0.0751,0.1433,0.0899,0.1468]; % neuode

% mMARE
% subplot(1,2,1);
boxchart(mMARE');
box on;
hold on;
plot(mean(mMARE,2),'-d');
hold off;
ylim([0.05,0.35]);
set(gca,'XTickLabel',{'PINOHI','Neural ODE + Mfg.','Calib. Ana.','Neural ODE'});
ylabel('Testing Average mMARE');
legend('Avg. mMARE Data','Avg. mMARE Mean');


% mMARE across scenarios
% x = 1:1:10;
% figure;
% plot(x,mMARE(1,:),'-sr',x,mMARE(2,:),'-sg',x,mMARE(3,:),'-sb',x,mMARE(4,:),'-sm');
% axis([0.5,10.5,0.05,0.3]);
% legend('PINOHI','Neural ODE + Mfg.','Calib. Ana.','Neural ODE');
% xlabel('Leave-One-Batch-Out Cross-Validation Scenario Index');
% ylabel('Testing Average mMARE');
