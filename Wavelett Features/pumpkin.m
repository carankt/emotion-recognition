% M = csvread('blueberry.csv',1,0);
% M = pumpkin;
shuffledArray = M(randperm(size(M,1)),:);
X = shuffledArray(:,8:71833);
Y = shuffledArray(:,1:7);
Xtrain = X(1:428,:);
Ytrain = Y(1:428,:);
Xtest = X(429:535,:);
Ytest = Y(429:535,:);
Xtrain = Xtrain';
Ytrain = Ytrain';
Xtest = Xtest';
Ytest = Ytest';
X = X';
Y = Y';

%load('blueberry.mat')
% Y = zeros(7,535);
% Y(1,1:127) = 1;
% Y(2,128:208) = 1;
% Y(3,209:254) = 1;
% Y(4,255:323) = 1;
% Y(5,324:394) = 1;
% Y(6,395:473) = 1;
% Y(7,474:535) = 1;
% Y';
% rng('default');
% hiddenSize1 = 200;
% autoenc1 = trainAutoencoder(Xtrain,hiddenSize1, ...
%     'MaxEpochs',200, ...
%     'L2WeightRegularization',0.004, ...
%     'SparsityRegularization',4, ...
%     'SparsityProportion',0.15, ...
%     'ScaleData', false);
% feat1 = encode(autoenc1,Xtrain);
% hiddenSize2 = 100;
% autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
%     'MaxEpochs',100, ...
%     'L2WeightRegularization',0.002, ...
%     'SparsityRegularization',4, ...
%     'SparsityProportion',0.1, ...
%     'ScaleData', false);
% feat2 = encode(autoenc2,feat1);
% softnet = trainSoftmaxLayer(feat2,Ytrain,'MaxEpochs',400);
% stackednet_two = stack(autoenc1,autoenc2,softnet);
% out = stackednet(Xtest);
% plotconfusion(Ytest,out)
% 




