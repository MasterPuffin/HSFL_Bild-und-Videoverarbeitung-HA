%% PROJEKT BIVI : Step 4 Alexnet

clc; %Kommandofenster bereinigen
clear; %Variablen bereinigen
close all; %alles schlieﬂen

%% 1. PLATE GENERATOR
% script1_PlateGenerator;

% Training Data
% Load images from Plate Generator into data store for augmentation step 2
imageDS = imageDatastore('AlexNetData','IncludeSubfolders',true,'LabelSource','foldernames');  % create DataStore
[trainingImageDS,validationImageDS] = splitEachLabel(imageDS,0.7,'randomized'); %70% als Trainingsdatn, 30% als Valodation aufteilen

%% 2. AUGMENTER

outputSize = [227 227 3];
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-10,10], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5], ...
    'RandScale',[0.9, 1.2], ...
    'RandXShear',[-10 10], ...
    'RandYShear',[-10 10]);

trainingImageAugDS = augmentedImageDatastore(outputSize, trainingImageDS, 'DataAugmentation',imageAugmenter);
validationImageAugDS = augmentedImageDatastore(outputSize, validationImageDS, 'DataAugmentation',imageAugmenter);


%% 3. AlexNet

net = alexnet; %HIER KOMMT ALEX, VORHANG AUF

%AlexNet letzten drei Layer ersetzen
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(trainingImageDS.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,...
        'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Trainingspotionen festlegen
options = trainingOptions('sgdm',...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...     
    'ValidationData',validationImageAugDS, ...
    'ValidationFrequency',3, ...
    'ValidationPatience', 5, ...     
    'Verbose',false, ...
    'Plots','training-progress');

%Alexnet trainieren, haupts‰chlich die letzten Schichten
netTransfer = trainNetwork(trainingImageAugDS,layers,options);

%Mit Testbildern testen
testDS = imageDatastore('PlatesCuttedFromPicAndLabelsAlex','IncludeSubfolders',true,'LabelSource','foldernames');
predictedLabels = classify(netTransfer, testDS);
accuracy = mean(predictedLabels == testDS.Labels)
