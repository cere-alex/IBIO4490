clear all ; close all; clc
conf.calDir=('./imageNet200'); % ubicacion archivos a utilizar    `
conf.numClasses = 200 ;
conf.numTrain = 15 ;
conf.numTest = 15 ;
classes = dir(fullfile(conf.calDir,'train')) ;
%classes = dir(fullfile(conf.calDir)) ;
classes = classes([classes.isdir]);
classes = {classes(3:conf.numClasses+2).name} ;
images = {} ;
imageClass = {} ;
% for ci = 1:length(classes)
%   ims = dir(fullfile(conf.calDir, classes{ci}, '*.JPEG'))' ;
%   ims = vl_colsubset(ims, conf.numTrain + conf.numTest) ;
%   ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false) ;
%   images = {images{:}, ims{:}} ;
%   imageClass{end+1} = ci * ones(1,length(ims)) ;
% end
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
conf.numSpatialX = 2 ;
conf.numSpatialY = 2 ;
conf.phowOpts = {'Step', 3} ;
model.classes = classes ;
model.phowOpts = conf.phowOpts ;
model.numSpatialX = conf.numSpatialX ;
model.numSpatialY = conf.numSpatialY ;
conf.quantizer = 'kdtree' ;
model.quantizer = conf.quantizer ;
model.vocab = [] ;
model.w = [] ;
model.b = [] ;
model.classify = @classify ;
%%
%parte del entrenamiento
conf.numWords = 600 ;
conf.prefix = 'baseline' ;
conf.dataDir = 'imageNet200/' ;
conf.vocabPath = fullfile(conf.dataDir, [conf.prefix '-vocab.mat']) ;
conf.clobber = false ;

if ~exist(conf.vocabPath) || conf.clobber

  % Get some PHOW descriptors to train the dictionary
  selTrainFeats = vl_colsubset(selTrain, 30) ;
  descrs = {} ;
  %for ii = 1:length(selTrainFeats)
  parfor ii = 1:length(selTrainFeats)
    im = imread(fullfile(conf.calDir,'train', images{selTrainFeats(ii)})) ;
    im = standarizeImage(im) ;
    [drop, descrs{ii}] = vl_phow(im, model.phowOpts{:}) ;
  end  

  descrs = vl_colsubset(cat(2, descrs{:}), 10e4) ;
  descrs = single(descrs) ;

  % Quantize the descriptors to get the visual words
  vocab = vl_kmeans(descrs, conf.numWords, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50) ;
  save(conf.vocabPath, 'vocab') ;
else
  load(conf.vocabPath) ;
end


%%
% comput especial histograma 
conf.histPath = fullfile(conf.dataDir, [conf.prefix '-hists.mat']) ;
model.vocab = vocab ;

if strcmp(model.quantizer, 'kdtree')
  model.kdtree = vl_kdtreebuild(vocab) ;
end

if ~exist(conf.histPath) || conf.clobber
  hists = {} ;
  parfor ii = 1:length(images)
  %for ii = 1:length(images)
    fprintf('Processing %s (%.2f %%)\n', images{ii}, 100 * ii / length(images)) ;
    if isfile(fullfile(conf.calDir,'train', images{ii}))
        im = imread(fullfile(conf.calDir,'train', images{ii})) ;
    else
        im = imread(fullfile(conf.calDir,'test', images{ii})) ;
    end
    hists{ii} = getImageDescriptor(model, im);
  end

  hists = cat(2, hists{:}) ;
  save(conf.histPath, 'hists') ;
else
  load(conf.histPath) ;
end

%%
%Computar caracteristicas del mapa

psix = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5) ;

%%
% train con svm
conf.modelPath = fullfile(conf.dataDir, [conf.prefix '-model.mat']) ;
conf.svm.solver = 'sdca' ;
conf.svm.C = 10 ;
conf.svm.biasMultiplier = 1 ;

if ~exist(conf.modelPath) || conf.clobber
  switch conf.svm.solver
    case {'sgd', 'sdca'}
      lambda = 1 / (conf.svm.C *  length(selTrain)) ;
      w = [] ;
      parfor ci = 1:length(classes)
        perm = randperm(length(selTrain)) ;
        fprintf('Training model for class %s\n', classes{ci}) ;
        y = 2 * (imageClass(selTrain) == ci) - 1 ;
        [w(:,ci) b(ci) info] = vl_svmtrain(psix(:, selTrain(perm)), y(perm), lambda, ...
          'Solver', conf.svm.solver, ...
          'MaxNumIterations', 50/lambda, ...
          'BiasMultiplier', conf.svm.biasMultiplier, ...
          'Epsilon', 1e-3);
      end

    case 'liblinear'
      svm = train(imageClass(selTrain)', ...
                  sparse(double(psix(:,selTrain))),  ...
                  sprintf(' -s 3 -B %f -c %f', ...
                          conf.svm.biasMultiplier, conf.svm.C), ...
                  'col') ;
      w = svm.w(:,1:end-1)' ;
      b =  svm.w(:,end)' ;
  end

  model.b = conf.svm.biasMultiplier * b ;
  model.w = w ;

  save(conf.modelPath, 'model') ;
else
  load(conf.modelPath) ;
end
%%
% Estima la clase de los test de imagenes
scores = model.w' * psix + model.b' * ones(1,size(psix,2)) ;
[drop, imageEstClass] = max(scores, [], 1) ;

%%
%Computa la matriz de confusion 
idx = sub2ind([length(classes), length(classes)], ...
              imageClass(selTest), imageEstClass(selTest)) ;
confus = zeros(length(classes)) ;
confus = vl_binsum(confus, ones(size(idx)), idx) ;

%%
% Plots

conf.resultPath = fullfile(conf.dataDir, [conf.prefix '-result']) ;

figure(1) ; clf;
subplot(1,2,1) ;
imagesc(scores(:,[selTrain selTest])) ; title('Scores') ;
set(gca, 'ytick', 1:length(classes), 'yticklabel', classes) ;
subplot(1,2,2) ;
imagesc(confus) ;
title(sprintf('Confusion matrix (%.2f %% accuracy)', ...
              100 * mean(diag(confus)/conf.numTest) )) ;
print('-depsc2', [conf.resultPath '.ps']) ;
save([conf.resultPath '.mat'], 'confus', 'conf') ;
