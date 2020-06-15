%% PROJEKT BIVI

clc; %Kommandofenster bereinigen
clear; %Variablen bereinigen
close all; %alles schließen

%% 1. PLATE GENERATOR
% script1_PlateGenerator;

% Training Data
% Load images from Plate Generator into data store for augmentation step 2
imageDS = imageDatastore('AlexNetData','IncludeSubfolders',true,'LabelSource','foldernames');  % create DataStore
%resizeImages;
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

%==== TO DO: Shear, Scale, Translation, Rotation müssen eingesetzt werden

trainingImageAugDS = augmentedImageDatastore(outputSize, trainingImageDS, 'DataAugmentation',imageAugmenter);
validationImageAugDS = augmentedImageDatastore(outputSize, validationImageDS, 'DataAugmentation',imageAugmenter);


%% 3. AlexNet

net = alexnet;

% Replace Final 3 Layers, Set the final fully connected layer to have 
%   the same size as the number of classes in the new data set 
%   (5, in this example). To learn faster in the new layers than 
%   in the transferred layers, increase the learning rate factors 
%   of the fully connected layer.

layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(trainingImageAugDS.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,...
        'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Train Network, Specify the training options, including mini-batch size 
%    and validation data. Set InitialLearnRate to a small value to 
%    slow down learning in the transferred layers. In the previous 
%    step, you increased the learning rate factors for the fully 
%    connected layer to speed up learning in the new final layers. 
%    This combination of learning rate settings results in fast learning 
%    only in the new layers and slower learning in the other layers.

options = trainingOptions('sgdm',...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...           % sehr klein -> untere(alte) Layer werden kaum gelernt
    'ValidationData',idmsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience', 5, ...        % 6 Inf wuerde den validation-stop ausschliessen
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(imdsTrain,layers,options);

% besser waere ein abschlissendes Testen mit´neuen´ Daten
%  und nicht mit den Validierungsdaten 
YPred = classify(netTransfer, idmsValidation);
accuracy = mean(YPred == idmsValidation.Labels)