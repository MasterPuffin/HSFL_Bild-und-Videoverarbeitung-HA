%% PROJEKT BIVI

clc; %Kommandofenster bereinigen
clear; %Variablen bereinigen
close all; %alles schließen

%% 1. PLATE GENERATOR
% script1_PlateGenerator;


%% 2. OWN AUGMENTER and Training Data

%Augmenter muss nur einmalig ausgeführt werden

% dsFL = imageDatastore('AlexNetData/FL');
% dsSL = imageDatastore('AlexNetData/SL');
% dsOther = imageDatastore('AlexNetData/Other');
% BiViAugmenter(dsFL, 'AlexNetDataAug/FL/');
% BiViAugmenter(dsSL, 'AlexNetDataAug/SL/');
% BiViAugmenter(dsOther, 'AlexNetDataAug/Other/');

augDataStore = imageDatastore('AlexNetDataAug','IncludeSubfolders',true,'LabelSource','foldernames');  % DataStore erstellen, Lables sind die Ordernamen
[trainingImageDS,validationImageDS] = splitEachLabel(augDataStore,0.7,'randomized'); %70% als Trainingsdaten, 30% als Valodation aufteilen


%% 2. AUGMENTER
%Der Matlab-Augmenter vervielfältig die Trainingsdaten durch Anwendung von
%Scherung, Rotation, Skalierung und Verschiebungen

outputSize = [227 227 3];
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-10,10], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5], ...
    'RandScale',[0.9, 1.2], ...
    'RandXShear',[-5 5], ...
    'RandYShear',[-5 5]);

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
    'MaxEpochs',1, ...
    'InitialLearnRate',1e-4, ...     
    'ValidationData',validationImageAugDS, ...
    'ValidationFrequency',3, ...
    'ValidationPatience', 20, ...     
    'Verbose',false, ...
    'Plots','training-progress');

%Alexnet trainieren, hauptsächlich die letzten Schichten
netTransfer = trainNetwork(trainingImageAugDS,layers,options);

%Mit Testbildern testen
testDS = imageDatastore('PlatesCuttedFromPicAndLabelsAlex','IncludeSubfolders',true,'LabelSource','foldernames');
predictedLabels = classify(netTransfer, testDS); %Netzt gegen neue Bilder testen
accuracy = mean(predictedLabels == testDS.Labels) %Accuracy auf Konsole
