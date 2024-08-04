clc;clear;
rng default 
load sourcemodel_suda1kn.mat; %load pre-trained source model
load hlabel_suda2kn.mat;%load Hard pseudo label
load slabel_suda2kn.mat;%load soft pseudo label 
imageInputSize = [227 227 3];
%load image
allImages = imageDatastore('.\3target\suda\2kn\',... 
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
for i=1:numel(allImages.Files)
    str=allImages.Files{i};
    num = regexp(str, '\d+', 'match');
    index2(i)=str2num(num{end});
end
label = categorical(label);
allImages.Labels=label(index2);
 [training_set,validation_set] = splitEachLabel(allImages,0.7,'randomized');    
%
augmented_training_set = augmentedImageSource(imageInputSize,training_set);
augmented_validation_set = augmentedImageSource(imageInputSize,validation_set);
for i=1:numel(augmented_training_set.Files)
    str=augmented_training_set.Files{i};
    num = regexp(str, '\d+', 'match');
    index(i)=str2num(num{end});
end

probabilitylabel1=probabilitylabel(index,:);
for i=1:numel(augmented_validation_set.Files)
    str=augmented_validation_set.Files{i};
    num = regexp(str, '\d+', 'match');
    index1(i)=str2num(num{end});
end

classes = categories(training_set.Labels);
numClasses = numel(classes);

%Hyperparameters
miniBatchSize = 32;
numEpochs = 30;
initialLearnRate = 1e-4;
decay = 0.01;
momentum = 0.9;
numObservationsTrain = numel(augmented_training_set.Files);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
averageGrad = [];
averageSqGrad = [];
numIterations = numEpochs * numIterationsPerEpoch;
monitor = trainingProgressMonitor(Metrics="Loss",Info="Epoch",XLabel="Iteration");

iteration = 0;
epoch = 0;
elapsedTime=[];

TTrain=training_set.Labels;
TTT=TTrain;
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;

    % Shuffle data.
    idxShuffle = randperm(numel(TTrain));
    i = 0;
    while i < numIterationsPerEpoch && ~monitor.Stop
        i = i + 1;
        iteration = iteration + 1;

        % Read mini-batch of data and convert the labels to dummy
        % variables.
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        idxMiniBatch = idxShuffle(idx);

        % Read mini-batch of data.
        tbl = readByIndex(augmented_training_set,idxMiniBatch);
        X1 = cat(4,tbl.input{:});

        T = zeros(numClasses, miniBatchSize,"single");
        for c = 1:numClasses
            T(c,TTrain(idxMiniBatch)==classes(c)) = 1;
        end

        % Convert mini-batch of data to a dlarray.
        X = dlarray(single(X1),"SSCB");

        % If training on a GPU, then convert data to a gpuArray.
        if canUseGPU
            X = gpuArray(X);
        end

        % Evaluate the model loss and gradients using dlfeval and the
        % modelLoss function.
         [loss,gradients,TX,probabilitylabel1,iiii] = dlfeval(@modelLossT,net,X,T, ...   %*********
             idxMiniBatch, probabilitylabel1);
%          
%         if epoch>3
            [~,TX1]=max(TX);
            TTrain(idxMiniBatch)=categorical(TX1');  
%         end
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);

        % Update the network parameters using the Adam optimizer.
        [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iteration,learnRate);

        % Update the training progress monitor.
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
        monitor.Progress = 100 * iteration/numIterations;
        %
elapsedTime = [elapsedTime,iiii];
elapsedTime1=sum(elapsedTime);
disp(['time：' num2str(elapsedTime1) 'second']);
    end
    TTT=[TTT,TTrain];
end

%% testing net
numOutputs = 1;
mbqTest = minibatchqueue(augmented_validation_set,numOutputs, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatchPredictors, ...
    MiniBatchFormat="SSCB");
load Datassuda_2kn.mat
YTest = modelPredictions(net,mbqTest,classes);
TTest = train_yy(index1);
accuracy = mean(TTest == double(YTest'));
figure
stem(TTest)
hold on
stem(double(YTest'))
legend('real','prediction')

%% plotconfusion
TTest1= categorical(TTest');
ptest=YTest;
figure
plotconfusion(ptest, TTest1)
PerItemAccuracy = mean(ptest == TTest1);
title(['overall per image accuracy ',num2str(round(100*PerItemAccuracy)),'%'])