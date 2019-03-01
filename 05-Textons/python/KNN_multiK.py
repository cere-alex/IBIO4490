#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:27:10 2019
@author: santiago
"""
import sys
sys.path.append('python')
import pickle
import numpy as np
from matplotlib import pyplot as plt
from computeTextons import computeTextons
from skimage import color
from skimage import io
from fbRun import fbRun
from assignTextons import assignTextons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix 

def unpickle(file):

    with open(file, 'rb') as fo:
        _dict = pickle.load(fo, encoding='latin1')
        _dict['labels'] = np.array(_dict['labels'])
        _dict['data'] = _dict['data'].reshape(_dict['data'].shape[0], 3, 32, 32).transpose(0,2,3,1)

    return _dict

def merge_dict(dict1, dict2):
    import numpy as np
    if len(dict1.keys())==0: return dict2
    new_dict = {key: (value1, value2) for key, value1, value2 in zip(dict1.keys(), dict1.values(), dict2.values())}
    for key, value in new_dict.items():
        if key=='data':
            new_dict[key] = np.vstack((value[0], value[1]))
        if key=='labels':
            new_dict[key] = np.hstack((value[0], value[1]))            
        elif key=='batch_label':
            new_dict[key] = value[1]
        else:
            new_dict[key] = value[0] + value[1]
    return new_dict

def get_data(data, sliced=1):
    from skimage import color
    import numpy as np
    data_x = data['data']
    data_x = color.rgb2gray(data_x)
    data_x = data_x[:int(data_x.shape[0]*sliced)]
    data_y = data['labels']
    data_y = data_y[:int(data_y.shape[0]*sliced)]
    return data_x, data_y

def load_cifar10(meta='cifar-10-batches-py', mode=1):
    assert mode in [1, 2, 3, 4, 5, 'test']
    _dict = {}
    import os
    if isinstance(mode, int):
        for i in range(mode):
            file_ = os.path.join(meta, 'data_batch_'+str(mode))           
            _dict = merge_dict(_dict, unpickle(file_))
    else:
        file_ = os.path.join(meta, 'test_batch')
        _dict = unpickle(file_)
    return _dict

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

#rr=np.random.randint(1,5)
rr=2
trainBatch=unpickle('cifar-10-python/cifar-10-batches-py/data_batch_'+str(rr))
testBatch=unpickle('cifar-10-python/cifar-10-batches-py/test_batch')
#testBatch=np.sort(testBatch)
#trainBatch=np.sort(trainBatch)

keys=list(trainBatch)

bl_train=trainBatch[keys[0]]
bl_test=testBatch[keys[0]]

lab_train=trainBatch[keys[1]]
lab_test=testBatch[keys[1]]

data_train=trainBatch[keys[2]]
data_test=testBatch[keys[2]]

filenames_train=trainBatch[keys[3]]
filenames_test=testBatch[keys[3]]

#Create a filter bank with deafult params
from fbCreate import fbCreate
sup=2
staSig=0.6
fb = fbCreate(sup,staSig) # fbCreate(**kwargs, vis=True) for visualization
    
#Load sample images from disk
    
imageAmount=1000
testImages=10000
allKs=[]
ACAs_train=[]
ACAs_test=[]

for fac in range(1,20):
    
    imgs=[]
    classes=[]
    idxx=0
    cl=0
    usedIdxs=[]
    
    while (len(classes)<imageAmount and cl<=9):
        
        if (lab_train[idxx]==cl and usedIdxs.count(idxx)==0): 
            actualImage=(data_train[idxx,:,:,:])
            imAux=color.rgb2gray(actualImage)
            imgs.append(imAux)
            classes.append(lab_train[idxx])
            usedIdxs.append(idxx)
            idx=0
            if classes.count(cl)==(imageAmount/10):
                cl=cl+1
        idxx=idxx+1
                
    stacks=imgs[0]
    for idx in range(1,imageAmount):
        stacks=np.hstack((stacks,imgs[idx]))
        filterResponses=fbRun(fb,stacks) 
        k=8*fac
                    
    #Compute textons from filter
    [map,textons]=computeTextons(filterResponses,k)
    #Load more images
    testImgs=[]
    for idx in range(testImages):
        imAux=color.rgb2gray(data_test[idx,:,:,:])
        testImgs.append(imAux)
        
        #Calculate texton representation with current texton dictionary
        

    tmapsTrain=[]
    tmapsTest=[]
    for idx in range(imageAmount):
        Test=assignTextons(fbRun(fb,imgs[idx]),textons.transpose())    
        tmapsTrain.append(Test)

    
    for idx in range(testImages):
        mapTest=assignTextons(fbRun(fb,testImgs[idx]),textons.transpose())    
        tmapsTest.append(mapTest)

    trainHists=[]
    testHists=[]
    for idx in range(imageAmount):
        auxHist=histc(tmapsTrain[idx].flatten(),np.arange(k))
        trainHists.append(auxHist)
        
    for idx in range(testImages):
        auxHist=histc(tmapsTest[idx].flatten(),np.arange(k))
        testHists.append(auxHist) 
    
    trainHists_2=[]
    for idx in range(imageAmount):
        trainHists_2.append(trainHists[idx])    
        trainHists_2[idx]=list(trainHists_2[idx])
        classes[idx]=int(classes[idx])
  
    n_nei=3
    neigh=KNeighborsClassifier(n_neighbors=n_nei)                           
    neigh.fit(trainHists_2,classes) 


    pred_tr=[]  
    for idx in range(imageAmount):
        auxiliarArray=np.array([trainHists_2[idx]])
        pred_tr.append(neigh.predict(auxiliarArray))
        pred_tr[idx]=int(pred_tr[idx])  
    
    pred_tes=[]
    anns=[]    
    for idx in range(testImages):
        auxiliarArray=np.array([testHists[idx]])
        pred_tes.append(neigh.predict(auxiliarArray))
        pred_tes[idx]=int(pred_tes[idx])
        anns.append(lab_test[idx])
        anns[idx]=int(anns[idx])
    
   
    confussionMat_train=confusion_matrix(classes,pred_tr)*(1/imageAmount)
    confussionMat_test=confusion_matrix(anns,pred_tes)*(1/testImages) 

    ACA_train=0
    ACA_test=0
    for idx in range(0,10):
        ACA_train=ACA_train+confussionMat_train[idx,idx]
        ACA_test=ACA_test+confussionMat_test[idx,idx]
    
    allKs.append(k)
    ACAs_train.append(ACA_train)
    ACAs_test.append(ACA_test)
	# Saves relevant information into a .pickle file
    info={}
    info['Model']=neigh
    info['confussionMatrixTrain']=confussionMat_train
    info['ACA_train']=ACA_train
    info['confussionMatrixTest']=confussionMat_test
    info['ACA_test']=ACA_test

    with open('info_k='+str(k)+'_KNN_'+'Neighbors:'+str(n_nei)+'_supp:'+str(sup)+'sigma:'+str(staSig)+'TotalImgs:'+str(imageAmount)+'.pickle','wb') as handle:
        pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
aux={}
aux['allKs']=allKs     
with open('allKs_RF.pickle','wb') as handle:
    pickle.dump(aux, handle, protocol=pickle.HIGHEST_PROTOCOL)
aux={}
aux['ACAs_train']=ACAs_train      
with open('ACAs_train.pickle','wb') as handle:
    pickle.dump(aux, handle, protocol=pickle.HIGHEST_PROTOCOL)
aux={}
aux['ACAs_test']=ACAs_test      
with open('ACAs_train.pickle','wb') as handle:
    pickle.dump(aux, handle, protocol=pickle.HIGHEST_PROTOCOL) 


