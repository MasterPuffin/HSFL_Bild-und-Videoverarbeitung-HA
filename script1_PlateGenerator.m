
clear all;

%% Parameter
nop = 20; % Anzahl Nummernschilder

%% Kennzeicheeelemente+alphamaske laden und in double konvertieren
[sheet, sheetmap, sheetalpha] = imread('platesheet.png');
sheet = double(sheet)/255; sheetalpha = double(sheetalpha)/255;
% sheetalpha = repmat(sheetalpha,3);

%% Thresholding
sheetth = sum(sheet.*sheetalpha,3)>0.1;
sheetth = imfill(sheetth,'holes');

%% BoundingBoxen mit regionprops erhalten
labels = bwlabel(sheetth,8);
% figure(1), imshow(labels);
s = regionprops(sheetth,'BoundingBox');
boxes = cat(1,s.BoundingBox);
boxes(:,[1 2]) = ceil(boxes(:,[1 2]));
boxes(:,[3 4]) = floor(boxes(:,[3 4]));

%% Zeichen-RegionIndex-Paare
zname = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p"...
    ,"q","r","s","t","u","v","w","x","y","z","ae","oe","ue","0","1","2","3"...
    ,"4","5","6","7","8","9","eu","plate","tuev1","tuev2","tuev3","tuev4"...
    ,"tuev5","tuev6","land1","land2","land3","land4","land5","land6"...
    ,"land7","land8","land9","land10","land11","land12","land13"...
    ,"land14","land15","land16", ""];

index = [1 13 20 24 28 32 36 40 44 48 52 57 60 2 15 21 25 29 33 37 41 45 49 53 58 61 54 56 62 3 16 22 26 30 34 38 42 46 50 8 4 6 11 17 7 12 18 9 14 19 23 27 31 35 39 43 47 51 55 59 63 64 65];
%% ColorLabeling for SegNet
labelname = ["Background" zname(1:39)];
colorMat = (dec2base(0:39,4)-'0') .* 85;

for i=1:nop
    % Beispiel Kennzeichen
    % Vor dem letzten Nummernblock muss ein "" eingefuegt werden.
    %text = ["f" "l" zname(randi([42 47],1)) "land15" zname(randi([1 26],[1 2])) "" zname(randi([30 39],[1 3]))]; %FL
    %text = ["s" "l" zname(randi([42 47],1)) "land15" zname(randi([1 26],[1 2])) "" zname(randi([30 39],[1 3]))]; %SL
    % Random Kennzeichens
    is = [randi([1 26],[1 2]) randi([42 47],1) randi([48 63],1) randi([1 26],[1 2]) 64 randi([30 39],[1 3])];
    text = zname(is); %RANDOM = Other
    savename = [text([1 2 5 6 8 9 10])];
    
    %% Hintergrundblech
    kennzeichen=zeros(200,1200,3);

    % Einfuegeposition
    xos = 1;
    yos = 1;

    % Eu-Schild
    box = boxes(index(find(zname=="eu")), :);
    x = sheet(box(2):box(2)+box(4)-1, box(1):box(1)+box(3)-1,:);
    xalpha = sheetalpha(box(2):box(2)+box(4)-1, box(1):box(1)+box(3)-1,:);
    kennzeichen(yos:yos+box(4)-1, xos:xos+box(3)-1,:) = ...
        (1-xalpha).*kennzeichen(yos:yos+box(4)-1, xos:xos+box(3)-1,:)...
        + ...
        xalpha.*x;
    xos = xos+box(3);

    % restliches Blech
    box = boxes(index(find(zname=="plate")), :);
    x = sheet(box(2):box(2)+box(4)-1, box(1):box(1)+box(3)-1,:);
    xalpha = sheetalpha(box(2):box(2)+box(4)-1, box(1):box(1)+box(3)-1,:);
    kennzeichen(yos:yos+box(4)-1, xos:xos+box(3)-1,:) = ...
        (1-xalpha).*kennzeichen(yos:yos+box(4)-1, xos:xos+box(3)-1,:)...
        + ...
        xalpha.*x;

    xos = xos+10;
    yos = yos+15;

    %% Kennzeichen generieren
    tuevy = -5; tuevx = 10;
    landy = 40; landx = 3;
    zeichenabstand = 7;
    space = 20;
    for i=1:numel(text)
        if text(i)== ""
            xos = xos + space;
        else
            box = boxes(index(find(zname==text(i))), :);
            x = sheet(box(2):box(2)+box(4)-1, box(1):box(1)+box(3)-1,:);
            xalpha = sheetalpha(box(2):box(2)+box(4)-1, box(1):box(1)+box(3)-1,:);

            if startsWith(text(i),'tuev')
                kennzeichen(yos+tuevy:yos+tuevy+box(4)-1, xos+tuevx:xos+tuevx+box(3)-1,:) = ...
                (1-xalpha).*kennzeichen(yos+tuevy:yos+tuevy+box(4)-1, xos+tuevx:xos+tuevx+box(3)-1,:)...
                + ...
                xalpha.*x;
            elseif startsWith(text(i),'land')
                kennzeichen(yos+landy:yos+landy+box(4)-1, xos+landx:xos+landx+box(3)-1,:) = ...
                (1-xalpha).*kennzeichen(yos+landy:yos+landy+box(4)-1, xos+landx:xos+landx+box(3)-1,:)...
                + ...
                xalpha.*x;
                xos = xos+box(3)+zeichenabstand;
            else
                kennzeichen(yos:yos+box(4)-1, xos:xos+box(3)-1,:) = ...
                (1-xalpha).*kennzeichen(yos:yos+box(4)-1, xos:xos+box(3)-1,:)...
                + ...
                xalpha.*x;
                xos = xos+box(3)+zeichenabstand;
            end
        end
    end
    kennzeichen(:, xos:xos+12,:) = kennzeichen(:, 1005:1017,:);
    kennzeichen(:, xos+12+1:end,:) = 0;
    fertig = kennzeichen(1:120, 1:xos+12,:);
    
    folder = char("plates/");
    if ~exist(folder, 'dir')
       mkdir(folder)
    end
    imwrite(fertig, char("plates/"+[strjoin(savename,"")+".png"]), 'PNG');
end
