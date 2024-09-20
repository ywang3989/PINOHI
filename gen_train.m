P = [2,7,9,10,13,15,16,19,20,25,26,27,33,34,35,46,47,48,52,53,58,59,60,64,65,66,70,71,72,76,77,82,83,84,88,89,90];
C = [28,29,30,36,39,40,49,50,51,55,56,57,61,62,63,67,68,69,73,74,75,79,80,81,85,86,87,91,92,93];
S = [5,6,11,12,18];
T60 = sort([randsample(P,33),randsample(S,3),randsample(C,24)]);
T50 = sort([randsample(P,28),randsample(S,3),randsample(C,19)]);
T40 = sort([randsample(P,23),randsample(S,3),randsample(C,14)]);
T30 = sort([randsample(P,16),randsample(S,3),randsample(C,11)]);
T20 = sort([randsample(P,8),randsample(S,2),randsample(C,10)]);
T10 = sort([randsample(P,4),randsample(S,1),randsample(C,5)]);

% print
train = '';
for i = 1:10
    train = strcat(train,', ',int2str(T10(i)));
end
train