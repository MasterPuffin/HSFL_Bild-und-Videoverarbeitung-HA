%% PROJEKT BIVI

clc; %Kommandofenster bereinigen
clear; %Variablen bereinigen
close all; %alles schlieﬂen

%% 1. PLATE GENERATOR
%script1_PlateGenerator;
%script2_TrafficImageGenerator;
%script3_TrafficResizeToSameSize;

% Training Data
% Load images from Plate Generator into data store for augmentation step 2
imageDS = imageDatastore('TrainingData','IncludeSubfolders',true,'LabelSource','foldernames');  % create DataStore
[trainingImageDS,validationImageDS] = splitEachLabel(imageDS,0.7,'randomized'); %70% als Trainingsdatn, 30% als Valodation aufteilen

%% 2. AUGMENTER
outputSize = [360 480 3];
imageAugmenter = imageDataAugmenter('RandRotation',[-50,50],'RandXTranslation',[-5 5], 'RandYTranslation',[-5 5]);
trainingImageAugDS = augmentedImageDatastore(outputSize, trainingImageDS, 'DataAugmentation',imageAugmenter);
validationImageAugDS = augmentedImageDatastore(outputSize, validationImageDS, 'DataAugmentation',imageAugmenter);


%% 3. OWN AUGMENTER OPTIONAL

%% 4. SET UP NETWORK

% ----- einfaches DeepLearning Netzwerk definieren

layers = [
    imageInputLayer([360 480 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm',...
    'MaxEpochs',3, ...                    
    'ValidationData', validationImageDS,...  % validationImageDS oder validationImageAugDS  ...
    'ValidationFrequency',30,...
    'Verbose',false,...
    'Plots','training-progress');

net = trainNetwork(trainingImageDS,layers,options);  % trainingImageDS oder trainingImageAugDS

predictedLabels = classify(net, validationImageDS);
accuracy = mean(predictedLabels == validationImageDS.Labels)

%% 5. TRAIN NETWORK

%% 6. TEST NETWORK

%% 7. SET UP ALEXNET OPTIONAL

%% 8. DATA OUTPUT