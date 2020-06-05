
%%%%%%%%%%%%%%%%%%%% Bildgenerator %%%%%%%%%%%
% Fuegt Nummernschilder mit randomisierten Positionen und Groessen in
% verschiedene Hintergrundbilder ein.
% Erzeugt dabei Colorlabels fuer die SegmanticSegmantation
% Beim Abspeichern erhalten die Bilder als Dateinamen die Zeichenkette des
% Nummernschildes.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;

% Parameter

% Classcolor in RGB
%              Backgound   Nummernschild
classcolors = {[0 0 0]    [255 255 255]};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

platesIds = imageDatastore('plates');
backgroundIds = imageDatastore('backgrounds');

for iBG=1:numel(backgroundIds.Files)
    
    bg_orig = readimage(backgroundIds,iBG);
    bg_orig = imresize(bg_orig, [360 480], 'bilinear'); % Inputgroesse des SegNets

    for iPL=1:numel(platesIds.Files)
        bg = bg_orig;
        [plate INFO] = readimage(platesIds, iPL);
        [path name ext]=fileparts(INFO.Filename);
    
        %% Ramdom Size
        r = randi([40 300]) * 0.15 /100;
        plate = imresize(plate, r, 'bilinear');

        % Konvertierung in uint8, da ansonsten beim Abspeichern der Classlabels im 
        % Float-Format Rundungsfehler entstehen, wodurch die
        % Classlabel-Farbwerte nicht exakt 255 oder 0 sind. Das SegNet
        % funktioniert nur mit exakten Werten!
        classlabel = uint8(zeros(size(bg)));
            
        % Rondom position
        m = floor(rand*(size(bg,1)-size(plate,1)));
        n = floor(rand*(size(bg,2)-size(plate,2)));

        % Einfuegen
        bg(1+m:m+size(plate,1), 1+n:n+size(plate,2), :) = plate;
        
        % BackgroundClassLabel
        classlabel(:,:,1) = classcolors{1}(1);
        classlabel(:,:,2) = classcolors{1}(2);
        classlabel(:,:,3) = classcolors{1}(3);
        
        % PlateClassLabel
        classlabel(1+m:m+size(plate,1), 1+n:n+size(plate,2), 1) = classcolors{2}(1);
        classlabel(1+m:m+size(plate,1), 1+n:n+size(plate,2), 2) = classcolors{2}(2);
        classlabel(1+m:m+size(plate,1), 1+n:n+size(plate,2), 3) = classcolors{2}(3);
        
%         figure(1); imshow(imresize(bg,0.5))
%         pause(1)
        
        folder = char("images/");
        if ~exist(folder, 'dir')
           mkdir(folder)
        end
        filename = char(name + "_" + num2str(iBG) + ".png");
        fname=fullfile(folder,filename);
        imwrite(bg,fname,'PNG');

        folder = char("labels/");
        if ~exist(folder, 'dir')
           mkdir(folder)
        end
        filename = char(name + "_" + num2str(iBG) + ".png");
        fname=fullfile(folder,filename);
        imwrite(classlabel,fname,'PNG');
    end
        
end
    
