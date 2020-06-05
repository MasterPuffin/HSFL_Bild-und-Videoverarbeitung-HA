%% PROJEKT BIVI

clc; %Kommandofenster bereinigen
clear; %Variablen bereinigen
close all; %alles schlieﬂen

%% 1. PLATE GENERATOR
script1_PlateGenerator;
script2_TrafficImageGenerator;
script3_TrafficResizeToSameSize;

% Training Data
% Load images from Plate Generator into data store for augmentation step 2
%imageDS = imageDatastore('Folder?','IncludeSubfolders',true,'LabelSource','foldernames');  % create DataStore
%[trainingImageDS,validationImageDS] = splitEachLabel(imageDS,0.7,'randomized'); %70% als Trainingsdatn, 30% als Valodation aufteilen

%% 2. AUGMENTER

%% 3. OWN AUGMENTER OPTIONAL

%% 4. SET UP NETWORK

% ----- einfaches DeepLearning Netzwerk definieren

layers = [
    imageInputLayer([28 28 1])
    
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
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];



%% 5. TRAIN NETWORK

%% 6. TEST NETWORK

%% 7. SET UP ALEXNET OPTIONAL

%% 8. DATA OUTPUT