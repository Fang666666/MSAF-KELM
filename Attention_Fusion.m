function [oa, aa,K, ua,ub]  = Attention_Fusion(TY, TVT, wholeY, U, num)

load('KELM_Data.mat')
class_num=max(image_gt(:));
[rows,cols] = size(image_gt);
kelm_num=num(1);
test_num=num(2);


SA = self_attention(TY);
SA = sum(SA);

OF_wholeYY = zeros(class_num, rows*cols);
for s= 1:kelm_num
    OF_wholeYY = OF_wholeYY+U(s,:)'.*SA(s).*wholeY{s,1};
end 
[~, whole_label_index_actual]=max(OF_wholeYY);
whole_predict_map = reshape(whole_label_index_actual,[rows,cols]);
whole_predict_map=label2color(whole_predict_map,'new');
figure
imagesc(whole_predict_map);
axis image
axis off


%% output fusion
TYY = zeros(class_num,test_num);

for i =1:kelm_num
    TYY=TYY+U(i,:)'.*SA(i).*TY{i,1};
end


%% Calculate the classification measure index
 TY{kelm_num+1,1}=TYY;
 label_expected = zeros(1,test_num);
 label_actual = zeros(1,test_num);
 MissClassificationRate_Testing=0;
 for i = 1 : test_num 
     [~, label_index_expected]=max(TVT(:,i));
     label_expected(1,i) = label_index_expected;
     for j=1:kelm_num+1
         TYY=TY{j,1};
         [~, label_index_actual(j)]=max(TYY(:,i));
     end
     label_actual(1,i) = mode(label_index_actual);
     if mode(label_index_actual)~=label_index_expected
         MissClassificationRate_Testing=MissClassificationRate_Testing+1;
     end
 end
 TestingAccuracy=1-MissClassificationRate_Testing/test_num

[oa, aa, K, ua]=confusion(label_expected,label_actual);
pre_vector = reshape(image_gt,[1,rows*cols]);
for i = 1:test_num
    pre_vector(testingIndexRandom(i)) = label_actual(i);
end

figure

image_gt=label2color(image_gt,'new');
imagesc(image_gt);
axis image
axis off
colormap('jet')


end
      
