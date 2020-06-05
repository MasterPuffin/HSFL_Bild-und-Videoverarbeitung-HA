clear all; close all;

%%%%%%%%%%%%%% Parameter %%%%%%%%%%%%%%%%%
n_target = 480;
m_target = 360;
aspectratio_target = n_target / m_target;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

images = imageDatastore('images');
classlabels = imageDatastore('labels');

if numel(images.Files)~=numel(classlabels.Files)
    error('labes and images need to contain the same number of elements')
end

for i = 1:numel(images.Files)
   [im im_INFO]= readimage(images, i);
   im = double(im)/255;
   [im_path im_name im_ext]=fileparts(im_INFO.Filename);

   [cl cl_INFO]= readimage(classlabels, i);
   cl = double(cl)/255;
   [cl_path cl_name cl_ext]=fileparts(cl_INFO.Filename);
   
   width = size(im,2);
   height = size(im,1);
   aspectratio = width / height;
   
   % expand
   if aspectratio > aspectratio_target
       % dann hoeher machen
       newheight = ceil(width/aspectratio_target);
       newwidth = width;
       offset = floor((newheight-height)/2);
       
       im_expanded = zeros(newheight,newwidth,3);
       im_expanded(1+offset:height+offset,1:width,:) = im;
       
       cl_expanded = zeros(newheight,newwidth,3);
       cl_expanded(1+offset:height+offset,1:width,:) = cl;
       
   elseif aspectratio < aspectratio_target
       % dann breiter machen
       newheight = height;
       newwidth = ceil(height*aspectratio_target);
       offset = floor((newwidth-width)/2);
       
       im_expanded = zeros(newheight,newwidth,3);
       im_expanded(1:height,1+offset:width+offset,:) = im;

       cl_expanded = zeros(newheight,newwidth,3);
       cl_expanded(1:height,1+offset:width+offset,:) = cl;
   else
       im_expanded = im;
       cl_expanded = cl;
   end

   %% Resize and save
   
   % image
   imresized = imresize(im_expanded, [m_target n_target], 'bilinear');
   folder = char("samesize_images/");
   if ~exist(folder, 'dir')
       mkdir(folder)
   end
   filename = char(im_name + ".png");
   fname=fullfile(folder,filename);
   imwrite(imresized,fname,'PNG');
   
   % label
   clresized = imresize(cl_expanded, [m_target n_target], 'nearest');
   folder = char("samesize_labels/");
   if ~exist(folder, 'dir')
       mkdir(folder)
   end
   filename = char(cl_name + ".png");
   fname=fullfile(folder,filename);
   imwrite(clresized,fname,'PNG');
end