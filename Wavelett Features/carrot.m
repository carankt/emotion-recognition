rng('default');
hiddenSize1 = 250;
autoenc1 = trainAutoencoder(Xtrain,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);
feat1 = encode(autoenc1,Xtrain);
hiddenSize2 = 150;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',200, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
feat2 = encode(autoenc2,feat1);
hiddenSize3 = 50;
autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
feat3 = encode(autoenc3,feat2);
softnet = trainSoftmaxLayer(feat3,Ytrain,'MaxEpochs',200);
stackednet_trainused = stack(autoenc1,autoenc2,autoenc3,softnet);
% out = stackednet(Xtest);
% plotconfusion(Ytest,out)
