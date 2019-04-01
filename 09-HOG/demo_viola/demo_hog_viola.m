%save('memory.mat','w','b','bboxes','confidences','image_ids','test_scn_path','srcFiles','saved_faceDetector')
clear all;close all;clc;
addpath(genpath(pwd));
load('memory.mat');
srcFiles = dir([fullfile(pwd,'images/demo') '\*.jpg']);
test_files = dir(fullfile(pwd,'images/demo', '*.jpg'));
num_test_images = randi([1 length(test_files)],1,10);
for i =num_test_images
    filename= strcat(srcFiles(i).folder,'/',srcFiles(i).name);
    aux_image_ids = image_ids(strcmp(image_ids,srcFiles(i).name));
    aux_confidences = confidences(strcmp(image_ids,srcFiles(i).name));
    aux_bboxes = bboxes(strcmp(image_ids,srcFiles(i).name));
    image = imread(filename);
    if(size(image,3) > 1)
        image = rgb2gray(image);
    end
    image = imresize(image,[900 900]);
    faceDetector = saved_faceDetector;%funcion implementada por vision
    fbox = step(faceDetector,image);
    [m,n] =  size(int32(fbox));
    fout = insertObjectAnnotation(image,'rectangle',fbox,'Face');
    fout = insertText(fout, [20 20], m, 'BoxOpacity', 1,'FontSize', 14);
    cur_test_image = imread( fullfile(pwd,'images/demo', test_files(i).name));
    cur_detections = strcmp(test_files(i).name, image_ids);
    cur_bboxes = bboxes(cur_detections,:);
    cur_confidences = confidences(cur_detections);
    figure,
    subplot(1,2,1);
    imshow(cur_test_image);
    title('HOG');
    hold on;
    num_detections = sum(cur_detections);
    for j = 1:num_detections
        bb = cur_bboxes(j,:);
        plot(bb([1 3 3 1 1]),bb([2 2 4 4 2]),'g:','linewidth',2);
    end
    subplot(1,2,2);imshow(fout);title('viola-jones algorithm');
    pause();
end
close all;