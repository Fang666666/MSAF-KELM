function [outputs] = self_attention(input)
%SELF_ATTENTION 

load('KELM_Data.mat')

%load('TY.mat')
%input=TY;
T=input;
M=cell2mat(T);
[mrc,md]=size(M);
ensemble_num = scale_num*3+1;%%%%%%%%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
mr=ensemble_num;
mc=num_class;
MM=reshape(M,mr,mc,md);
MMM=permute(MM,[2,1,3]);
MMMM=cell(mc,1);
for i=1:mc
    MMMM{i,1}=reshape(MMM(i,:,:),mr,md);
end
for i=1:mc
    [~,S,~]=svd(MMMM{i,:},'econ');
    x(i,:)=diag(S)';
end
[input_num,input_length] = size(x);


%w_key = [0,0,1;1,1,0;0,1,0;1,1,0];
w_key = fspecial('gaussian',input_length,input_num);
%w_query = [1,0,1;1,0,0;0,0,1;0,1,1];
w_query = fspecial('gaussian',input_length,input_num);
%w_value = [0,2,0;0,3,0;1,0,3;1,1,0];
w_value = fspecial('gaussian',input_length,input_num);


keys = x * w_key;
querys = x * w_query;
values = x * w_value;



attn_scores = querys * keys';
% tensor([[ 2.,  4.,  4.],  # attention scores from Query 1
%         [ 4., 16., 12.],  # attention scores from Query 2
%         [ 4., 12., 10.]]) # attention scores from Query 3

attn_scores_softmax = ReLU(attn_scores);
% For readability, approximate the above as follows
%attn_scores_softmax = [[0.0, 0.5, 0.5], [0.0, 1.0, 0.0], [0.0, 0.9, 0.1]]
%attn_scores_softmax = [0.0, 0.5, 0.5; 0.0, 1.0, 0.0; 0.0, 0.9, 0.1];


attn_scores_softmax = attn_scores_softmax';
values1(:,1,:) = values;
attn_scores_softmax1(:,:,1) = attn_scores_softmax;
weighted_values = values1 .* attn_scores_softmax1;
weighted_values1 = permute(weighted_values,[2,1,3]);

outputs = sum(weighted_values1,2);
% tensor([[2.0000, 7.0000, 1.5000], [2.0000, 8.0000, 0.0000], [2.0000, 7.8000, 0.3000]]) # Output1、2、3
outputs = reshape(outputs,input_num,input_length)*0.0001;


