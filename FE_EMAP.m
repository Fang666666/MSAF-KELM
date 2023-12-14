function Feature_E = FE_EMAP(data,data_gt)

% init
[rows,cols] = size(data_gt);
d_pca = 7;
Dp = PCA(data,d_pca);
%% Spatial features

%% Compute EAP

%% EAP area
attr = 'area';
lambdas = [100 500 1000 5000];
EAP = ext_attribute_profile(reshape(Dp, rows, cols, d_pca), attr, lambdas);

EAP_a = double(reshape(EAP, rows*cols, size(EAP,3)));

%% EAP inertia
attr = 'inertia';
lambdas = [0.2 0.3 0.4 0.5];
EAP = ext_attribute_profile(reshape(Dp, rows, cols, d_pca), attr, lambdas);

EAP_i = double(reshape(EAP, rows*cols, size(EAP,3)));

%% EAP std
attr = 'std';
lambdas = [20 30 40 50];
EAP = ext_attribute_profile(reshape(Dp, rows, cols, d_pca), attr, lambdas);

EAP_s = double(reshape(EAP, rows*cols, size(EAP,3)));

%% EMAP
EMAP = [EAP_a, EAP_i, EAP_s];


npca = 30;
EMAP = PCA(EMAP,20);
Feature_E = EMAP;


end

