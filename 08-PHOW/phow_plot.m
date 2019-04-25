clear all; clc ; close all;

load('D:\Ale\University\UNIANDES\2do semestre\Vision Artificial\work-space\08-PHOW\imageNet200\baseline_100_image-hists.mat')
load('D:\Ale\University\UNIANDES\2do semestre\Vision Artificial\work-space\08-PHOW\imageNet200\baseline_100_image-model.mat')
load('D:\Ale\University\UNIANDES\2do semestre\Vision Artificial\work-space\08-PHOW\imageNet200\baseline_100_image-result.mat')
load('D:\Ale\University\UNIANDES\2do semestre\Vision Artificial\work-space\08-PHOW\imageNet200\baseline_100_image-vocab.mat')

classes = model.classes;

images = {} ;
imageClass = {} ;
for ci = 1:length(classes)
  ims_train= dir(fullfile(conf.calDir,'train', classes{ci}, '*.JPEG'))' ;
  ims_test= dir(fullfile(conf.calDir,'test', classes{ci}, '*.JPEG'))' ;
  ims_train = vl_colsubset(ims_train, conf.numTrain) ;
  ims_test = vl_colsubset(ims_test, conf.numTest) ;
  ims = cat(2,ims_train,ims_test);
  ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false) ;
  images = {images{:}, ims{:}} ;
  imageClass{end+1} = ci * ones(1,length(ims)) ;
end
selTrain = find(mod(0:length(images)-1, conf.numTrain+conf.numTest) < conf.numTrain) ;
selTest = setdiff(1:length(images), selTrain) ;
imageClass = cat(2, imageClass{:}) ;
model.vocab = vocab ;
if strcmp(model.quantizer, 'kdtree')
  model.kdtree = vl_kdtreebuild(vocab) ;
end

psix = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5) ;
scores = model.w' * psix + model.b' * ones(1,size(psix,2)) ;

[drop, imageEstClass] = max(scores, [], 1) ;
idx = sub2ind([length(classes), length(classes)], ...
              imageClass(selTest), imageEstClass(selTest)) ;
confus = zeros(length(classes)) ;
confus = vl_binsum(confus, ones(size(idx)), idx) ;

% Plots
figure(1) ; clf;
subplot(1,2,1) ;
imagesc(scores(:,[selTrain selTest])) ; title('Scores') ;
set(gca, 'ytick', 1:length(classes), 'yticklabel', classes) ;
subplot(1,2,2) ;
imagesc(confus) ;
title(sprintf('Confusion matrix (%.2f %% accuracy)', ...
              100 * mean(diag(confus)/conf.numTest) )) ;
print('-depsc2', [conf.resultPath '.ps']) ;
