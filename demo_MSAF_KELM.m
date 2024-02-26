clear, clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%  Load paths and datasets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('./DataSets/'))
addpath(genpath('./Self_Attention/'))
addpath(genpath('./Tools/'))
load Salinas_corrected.mat
load Salinas_gt.mat
load Feature_G.mat   % 3D-Gabor Feature
load Feature_E_L.mat % EMAP and LBP Feature

image = salinas_corrected;
image_gt = salinas_gt;
[rows,cols,depth] = size(image);

Res = 3.7;
scale_num = 3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%  Get Superpixel segmentation label
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labels_superpixel = Superpixel_segmentation(image, image_gt, scale_num, Res);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%  Train Test index
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
training_rate=0.005;
% training_num=15
num_class=max(max(image_gt));
trainingIndexRandom=cell(num_class,1);
gt_flatten = reshape(image_gt,rows*cols,1);
for i =1:num_class
    index = find(gt_flatten==i);
    
    rndIDX = randperm(length(index));
    train_num_class_i = round(training_rate * length(index));
    % train_num_class_i=repelem(training_num,num_class);
    trainingIndexRandom{i,1}= index(rndIDX(1:train_num_class_i));
end

trainingIndexRandom=cell2mat(trainingIndexRandom);
rndIDXt = randperm(length(trainingIndexRandom));
trainingIndexRandom = trainingIndexRandom(rndIDXt);
[trainingIndexRandom_rows,trainingIndexRandom_cols] = ind2sub(size(image_gt),trainingIndexRandom);
wholeSample = find(gt_flatten~=0);
testingIndexRandom=setdiff(wholeSample,trainingIndexRandom);
[testingIndexRandom_rows,testingIndexRandom_cols] = ind2sub(size(image_gt),testingIndexRandom);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%  Origin Spectral Feature, EMAP Feature, LBP Feature, 3D-Gabor Feature Extration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha=10;
train_loc = trainingIndexRandom;
train_loc_rows = trainingIndexRandom_rows;
train_loc_cols = trainingIndexRandom_cols;
test_loc = testingIndexRandom;
test_loc_rows = testingIndexRandom_rows;
test_loc_cols = testingIndexRandom_cols;
labeled_loc = [trainingIndexRandom;testingIndexRandom];
whole_loc = double(1:rows*cols)';

train_y = gt_flatten(train_loc)';
test_y = gt_flatten(test_loc)';
labeled_y = gt_flatten(labeled_loc)';
scale_num = size(labels_superpixel,1);
ensemble_num = scale_num*3+1;
train_num = size(train_y,2);
test_num = size(test_y,2);
labeled_num = size(labeled_y,2);


Data = image;
for i=1:depth
    Data(:,:,i) = (Data(:,:,i)-min(min(Data(:,:,i))))/ (max(max(Data(:,:,i)))-min(min(Data(:,:,i))));
end

spectral_data=reshape(Data,rows*cols,depth);
train_data_spectral = spectral_data(trainingIndexRandom,:);
test_data_spectral = spectral_data(testingIndexRandom,:);
whole_data_spectral = spectral_data;

superpixel_data = cell(scale_num,1);
soi_index = cell(scale_num,1);
for k=1:scale_num
    superpixel_data{k,1}=zeros(rows*cols,depth);
end

for k=1:scale_num
    soi_index{k,1} = labels_superpixel{k,1}(whole_loc);
    soi_index{k,1}  = unique(soi_index{k,1});
end


for k = 1:scale_num
    for j=1:size(soi_index{k,1},1)
        super_index = find(labels_superpixel{k,1}==soi_index{k,1}(j,1));
        superpixel_data{k,1}(super_index,:) = repmat(mean(spectral_data(super_index,:)),size(super_index ,1),1);
    end
end

for k = 1:scale_num
    train_data_EMAP{k,1}=zeros(train_num,depth);
    test_data_EMAP{k,1}=zeros(test_num,depth);
    whole_data_EMAP{k,1}=zeros(rows*cols,depth);
    train_data_LBP{k,1}=zeros(test_num,depth);
    test_data_LBP{k,1}=zeros(test_num,depth);
    whole_data_LBP{k,1}=zeros(rows*cols,depth);
    train_data_Gabor{k,1}=zeros(test_num,depth);
    test_data_Gabor{k,1}=zeros(test_num,depth);
    whole_data_Gabor{k,1}=zeros(rows*cols,depth);
    
    train_data_EMAP{k,1} = Feature_E{k,1}(trainingIndexRandom,:);
    test_data_EMAP{k,1} = Feature_E{k,1}(testingIndexRandom,:);
    whole_data_EMAP{k,1} = Feature_E{k,1}(whole_loc,:);
    train_data_LBP{k,1} = Feature_L{k,1}(trainingIndexRandom,:);
    test_data_LBP{k,1} = Feature_L{k,1}(testingIndexRandom,:);
    whole_data_LBP{k,1} = Feature_L{k,1}(whole_loc,:);
    train_data_Gabor{k,1} = Feature_G{k,1}(trainingIndexRandom,:);
    test_data_Gabor{k,1} = Feature_G{k,1}(testingIndexRandom,:);
    whole_data_Gabor{k,1} = Feature_G{k,1}(whole_loc,:);
end

train_x = cell(ensemble_num,1);
train_x{1,1}=train_data_spectral;

for i =1:scale_num
    train_x{i+1,1}=train_data_LBP{i,1};
end


for j = 1:scale_num
    train_x{scale_num+1+j,1}=train_data_EMAP{j,1};
end


for k = 1:scale_num
    train_x{2*scale_num+1+k,1}=train_data_Gabor{k,1};
end


test_x = cell(ensemble_num,1);
test_x{1,1}=test_data_spectral;

for i =1:scale_num
    test_x{i+1,1}=test_data_LBP{i,1};
end


for j = 1:scale_num
    test_x{scale_num+1+j,1}=test_data_EMAP{j,1};
end

for k = 1:scale_num
    test_x{2*scale_num+1+k,1}=test_data_Gabor{k,1};
end

whole_x = cell(ensemble_num,1);
whole_x{1,1}=whole_data_spectral;

for i =1:scale_num
    whole_x{i+1,1}=whole_data_LBP{i,1};
end


for j = 1:scale_num
    whole_x{scale_num+1+j,1}=whole_data_EMAP{j,1};
end


for k = 1:scale_num
    whole_x{2*scale_num+1+k,1}=whole_data_Gabor{k,1};
end

save KELM_Data.mat testingIndexRandom train_x train_y test_x test_y whole_x image_gt num_class scale_num

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%  KELM With Attention Fusion Strategy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear 

[TY, TVT, wholeY, U, num]=DKELM([1,100,10000], 3, 'RBF_kernel',[1e2, 1e5, 1e7]);
[oa, aa, K, ua]=Attention_Fusion(TY, TVT, wholeY, U, num);
result_ae= [ua;oa;aa;K];