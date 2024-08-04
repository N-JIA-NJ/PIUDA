clc;clear;
rng default 
load alexnet.mat; 
imageInputSize = [227 227 3];
%load image
allImages = imageDatastore('.\2source\suda\3kn\',...  
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
 [training_set,validation_set] = splitEachLabel(allImages,0.7,'randomized');  
%
augmented_training_set = augmentedImageSource(imageInputSize,training_set);
augmented_validation_set = augmentedImageSource(imageInputSize,validation_set);

%build network
classes = categories(training_set.Labels);
numClasses = numel(classes);
layers=layers_1(1:end-1);
net = dlnetwork(layers);

%Hyperparameters
miniBatchSize = 32;
numEpochs = 30;
initialLearnRate = 1e-4;
decay = 0.01;
momentum = 0.9;

%
numObservationsTrain = numel(augmented_training_set.Files);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
averageGrad = [];
averageSqGrad = [];
numIterations = numEpochs * numIterationsPerEpoch;
monitor = trainingProgressMonitor(Metrics="Loss",Info="Epoch",XLabel="Iteration");

%
iteration = 0;
epoch = 0;

while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;

    % Shuffle data.
    TTrain=training_set.Labels;
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
        T1 = tbl.response;

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
         [loss,gradients] = dlfeval(@modelLoss1,net,X,T,idxMiniBatch);

               
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);

        % Update the network parameters using the Adam optimizer.
        [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iteration,learnRate);

        % Update the training progress monitor.
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
        monitor.Progress = 100 * iteration/numIterations;
    end
end

%% testing net
numOutputs = 1;
mbqTest = minibatchqueue(augmented_validation_set,numOutputs, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatchPredictors, ...
    MiniBatchFormat="SSCB");

YTest = modelPredictions(net,mbqTest,classes);
TTest = validation_set.Labels;
accuracy = mean(TTest == YTest);
figure
confusionchart(TTest,YTest);


