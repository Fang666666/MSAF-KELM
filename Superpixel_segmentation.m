function labels_superpixel = Superpixel_segmentation(image, image_gt, scale_num, Res)
clc

delta = 0.7;
labels_superpixel = cell(scale_num,1);
[rows, cols]=size(image_gt);
S = floor(100*delta^(Res^(1/2)));
N = zeros(1,scale_num);
for i =1:scale_num
    N(1,i)= floor(rows*cols/(2^(i-1)*S));
end
%% PCA one
[rows,cols,depts] = size(image);
X = reshape(image,rows*cols,depts);
z = zscore(X);
sigma = std(z);
Sigma = cov(z);
[V D] = eig(Sigma);
[D,ind] = sort(diag(D),'descend');
V = V(:,ind);
DR = 0.95;
d_pca=find((cumsum(D)/sum(D))>DR,1); % 
Dp = z*V(:,d_pca);
PC1_z = Dp;
PC1_z = reshape(PC1_z,rows,cols);
grey_img = PC1_z;

for s =1:scale_num
    nC = N(1,s);
    %// nC is the target number of superpixels.
    %// Call the mex function for superpixel segmentation\
    %// !!! Note that the output label starts from 0 to nC-1.
    t = cputime;
    lambda_prime = 0.5;sigma = 5.0; 
    conn8 = 1; % flag for using 8 connected grid graph (default setting).
    [labels] = mex_ers(double(grey_img),nC);
    fprintf(1,'Use %f sec. \n',cputime-t);
    fprintf(1,'\t to divide the image into %d superpixels.\n',nC);
    labels_superpixel{s,1} = labels;
     
end
end


