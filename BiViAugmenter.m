%%%%%%%%%%%%%%%%%%%% BiViAugmenter %%%%%%%%%%%
% Generiert je Bild in ds 100 Bilder mit unterschiedlichen realistischen
% Farbtönen und Helligkeiten
% ds: ein DataStore mit Bildern
% p: relativer Zielpfad (String)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function BiViAugmenter(ds, p)
    for i=1:numel(ds.Files)
        [file, info] = readimage(ds, i);
        for a=1:100
            f = file;
            increase = randi([0 20]);
            color = randi([1 3]);
            f(:,:,color) = f(:,:,color) + increase;
            f(:,:,color) = min(f(:,:,color),255);
            %augmented1 = jitterColorHSV(file,'Hue',[0.05 1]);
            augmented = jitterColorHSV(f,'Brightness',[-0.6 0]);
            [~, name, ~]=fileparts(info.Filename);
            imwrite(augmented, fullfile(p, char(name + "_aug" + num2str(a) + ".png")));
        end
    end
end