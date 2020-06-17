%% PROJEKT BIVI

clc; %Kommandofenster bereinigen
clear; %Variablen bereinigen
close all; %alles schließen

%% 1. PLATE GENERATOR
% script1_PlateGenerator;


%% 2. OWN AUGMENTER and Training Data
%Augmenter muss nur einmalig ausgeführt werden

%Wir erstellen drei Datastores zur Weiterverarbeitung und speichern diese
% in einem neuen Datastore ab
%lesen:
%dsFL = imageDatastore('TrainingData2/FL');
%dsSL = imageDatastore('TrainingData2/SL');
%dsOther = imageDatastore('TrainingData2/Other');

%Die Bilder durch den BiVI Augmenter hauen und im entspechenden Verzeichnis
%abspeichern:
%BiViAugmenter(dsFL, 'TrainingDataAug/FL/');
%BiViAugmenter(dsSL, 'TrainingDataAug/SL/');
%BiViAugmenter(dsOther, 'TrainingDataAug/Other/');

%Datastore neu erstellen:
augDataStore = imageDatastore('TrainingDataAug','IncludeSubfolders',true,'LabelSource','foldernames');  % DataStore erstellen, Lables sind die Ordernamen
[trainingImageDS,validationImageDS] = splitEachLabel(augDataStore,0.7,'randomized'); %70% als Trainingsdaten, 30% als Valodation aufteilen
%% 2. AUGMENTER
%Der Matlab-Augmenter vervielfältig die Trainingsdaten durch Anwendung von
%Scherung, Rotation, Skalierung und Verschiebungen

outputSize = [50 200 3];
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-10,10], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5], ...
    'RandScale',[0.9, 1.2], ...
    'RandXShear',[-5 5], ...
    'RandYShear',[-5 5]);
%Augmented Data Stores erstellen
trainingImageAugDS = augmentedImageDatastore(outputSize, trainingImageDS, 'DataAugmentation',imageAugmenter);
validationImageAugDS = augmentedImageDatastore(outputSize, validationImageDS, 'DataAugmentation',imageAugmenter);


%% 4. SET UP NETWORK

% ----- einfaches DeepLearning Netzwerk definieren

layers = [
    imageInputLayer([50 200 3])
    
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
    'MaxEpochs',2, ...                    
    'ValidationData', validationImageAugDS,...  
    'ValidationFrequency',10,...
    'Verbose',false,...
    'MiniBatchSize', 10, ...    
    'Plots','training-progress');

net = trainNetwork(trainingImageAugDS,layers,options);  %das Netzwerk wird trainiert


%% 6. TEST NETWORK

testDS = imageDatastore('PlatesCuttedFromPicAndLabels','IncludeSubfolders',true,'LabelSource','foldernames'); %Testdaten in einen Datastore laden
predictedLabels = classify(net, testDS); %Netz testen
accuracy = mean(predictedLabels == testDS.Labels) %Erkennungsrate ausgeben auf die Konsole
