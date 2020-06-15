%% PROJEKT BIVI

clc; %Kommandofenster bereinigen
clear; %Variablen bereinigen
close all; %alles schlie�en

%% 1. PLATE GENERATOR
% script1_PlateGenerator;
% script2_TrafficImageGenerator;
% script3_TrafficResizeToSameSize;

% Training Data
% Load images from Plate Generator into data store for augmentation step 2
imageDS = imageDatastore('PlatesCuttedFromPicAndLabels','IncludeSubfolders',true,'LabelSource','foldernames');  % create DataStore
[trainingImageDS,validationImageDS] = splitEachLabel(imageDS,0.7,'randomized'); %70% als Trainingsdatn, 30% als Valodation aufteilen

%% 2. AUGMENTER

outputSize = [200 50 3];
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-10,10], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5], ...
    'RandScale',[0.9, 1.2], ...
    'RandXShear',[-10 10], ...
    'RandYShear',[-10 10]);

%==== TO DO: Shear, Scale, Translation, Rotation m�ssen eingesetzt werden

trainingImageAugDS = augmentedImageDatastore(outputSize, trainingImageDS, 'DataAugmentation',imageAugmenter);
validationImageAugDS = augmentedImageDatastore(outputSize, validationImageDS, 'DataAugmentation',imageAugmenter);

%% 3. OWN AUGMENTER OPTIONAL

%% 4. SET UP NETWORK

% ----- einfaches DeepLearning Netzwerk definieren

layers = [
    imageInputLayer([200 50 3])
    
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
    classificationLayer
];


%% 5. TRAIN NETWORK

% 'ExecutionEnvironment' 'parallel' for gpu, 'cpu' for cpu training
options = trainingOptions('sgdm',...
    'MaxEpochs',20, ...                    
    'ValidationData', validationImageAugDS,...  % validationImageDS oder validationImageAugDS  ...
    'ValidationFrequency',5,...
    'Verbose',false,...
    'MiniBatchSize', 10, ...    
    'ExecutionEnvironment', 'parallel', ...
    'Plots','training-progress');

net = trainNetwork(trainingImageAugDS,layers,options);  % trainingImageDS oder trainingImageAugDS


%% 6. TEST NETWORK

testDS = imageDatastore('PlatesCuttedFromPicAndLabels','IncludeSubfolders',true,'LabelSource','foldernames');

outputSize = [200 50 3]; %only for resizing :D
imageAugmenter2 = imageDataAugmenter();
testDSsize = augmentedImageDatastore(outputSize, testDS, 'DataAugmentation',imageAugmenter2);

predictedLabels = classify(net, testDSsize);
accuracy = mean(predictedLabels == testDS.Labels)
%% 7. SET UP ALEXNET OPTIONAL

%% 8. DATA OUTPUT