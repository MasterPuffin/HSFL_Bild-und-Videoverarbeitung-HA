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
            shadow = randi([0 1]); %Schatten ja oder nein per Zufall
            increase = randi([0 20]); % Zufallszahl für Farbwertaddition
            color = randi([1 3]); % Zufalls-Farbindex 1, 2, 3   r, g, b
            f(:,:,color) = f(:,:,color) + increase; % Zufallszahl auf Zufallsfarbwert addieren
            f(:,:,color) = min(f(:,:,color),255); % auf 255 begrenzen, damit es ein valider RGB Wert bleibt
            %augmented1 = jitterColorHSV(file,'Hue',[0.05 1]);
            augmented = jitterColorHSV(f,'Brightness',[-0.5 0]); % Zufällige Helligkeit
                if (shadow==1)
                    augmented(1:floor(end/2),:,:) = jitterColorHSV(augmented(1:floor(end/2),:,:),'Brightness',[-0.3 0]); % obere Bildhälfte wird verdunkelt
                end
            [~, name, ~]=fileparts(info.Filename);
            imwrite(augmented, fullfile(p, char(name + "_aug" + num2str(a) + ".png"))); % Datei speichern
        end
    end
end