%% PROJEKT BIVI

clc; %Kommandofenster bereinigen
clear; %Variablen bereinigen
close all; %alles schlieﬂen

%% 1. PLATE GENERATOR
% script1_PlateGenerator;

%% Training Data
% Bilder zum Training in einen DataStore laden 
imageDS = imageDatastore('TrainingData2','IncludeSubfolders',true,'LabelSource','foldernames');  % DataStore erstellen, Lables sind die Ordernamen
[trainingImageDS,validationImageDS] = splitEachLabel(imageDS,0.7,'randomized'); %70% als Trainingsdaten, 30% als Valodation aufteilen

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
    'MaxEpochs',20, ...                    
    'ValidationData', validationImageDS,...  % validationImageDS oder validationImageAugDS  ...
    'ValidationFrequency',5,...
    'Verbose',false,...
    'MiniBatchSize', 10, ...    
    'Plots','training-progress');

net = trainNetwork(trainingImageDS,layers,options);  %das Netzwerk wird trainiert


%% 6. TEST NETWORK

testDS = imageDatastore('PlatesCuttedFromPicAndLabels','IncludeSubfolders',true,'LabelSource','foldernames'); %Testdaten in einen Datastore laden
predictedLabels = classify(net, testDS); %Netz testen
accuracy = mean(predictedLabels == testDS.Labels) %Erkennungsrate ausgeben auf die Konsole
