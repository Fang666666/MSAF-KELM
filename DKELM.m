function [TY, TVT, wholeY, U, num] = DKELM (Regularization_coefficient,layers,Kernel_type, kernel_para)
%%%%%%%%%%% Load training dataset
load('KELM_Data.mat')
class_num=max(image_gt(:));
[rows,cols] = size(image_gt);
%% Initialization
C = Regularization_coefficient;

train_num=size(train_y,2);
test_num=size(test_y,2);
T = train_y;%(1,240)
TVT = test_y; %(1,10009)

%% Preprocessing the data of classification
sorted_target=sort(cat(2,T,TVT),2);    
label=zeros(1,1);  % Find and save in 'label' class label from training and testing data sets
label(1,1)=sorted_target(1,1);
j=1;
for i = 2:(train_num+test_num)
    if sorted_target(1,i) ~= label(1,j)
        j=j+1;
        label(1,j) = sorted_target(1,i);
    end
end                                     
%% Processing the targets of training 
temp_T=zeros(class_num, train_num);

for i = 1:train_num
    for j = 1:class_num
        if label(1,j) == T(1,i)
            break;
        end
    end
    temp_T(j,i)=1;                      
end
T=temp_T*2-1;                           
%% Processing the targets of testing
temp_TV_T=zeros(class_num, test_num);

for i = 1:test_num
    for j = 1:class_num
        if label(1,j) == TVT(1,i)
            break;
        end
    end
    temp_TV_T(j,i)=1;                   
end
TVT=temp_TV_T*2-1;                     
%% Training Phase 
kelm_num = size(train_x,1);
temp_inputLayer = cell(kelm_num,1);
OutputWeight_final = cell(kelm_num,1);
OutputWeight = cell(kelm_num,layers-1);
Y = cell(kelm_num,1);   

for s = 1:kelm_num
    temp_inputLayer{s,1} =zscore(train_x{s,1}');    
    temp = kernel_matrix(temp_inputLayer{s,1}',Kernel_type, kernel_para(1));
    OutputWeight{s,1}=(temp+speye(train_num)/C(1))\temp_inputLayer{s,1}';
    temp_inputLayer{s,1} =  (OutputWeight{s,1}*temp_inputLayer{s,1});
%%  
    if layers>=3
        for i = 2:(layers-1)
            temp = kernel_matrix(temp_inputLayer{s,1}',Kernel_type, kernel_para(i));
            OutputWeight{s,i} = (temp+speye(train_num)/C(i))\temp_inputLayer{s,1}';
            temp_inputLayer{s,1} =  (OutputWeight{s,i}*temp_inputLayer{s,1});
        end
        temp = kernel_matrix(temp_inputLayer{s,1}',Kernel_type, kernel_para(layers));
        OutputWeight_final{s,1}=((temp+speye(train_num)/C(layers))\(T'));
        Y{s,1} = OutputWeight_final{s,1}'*temp;                             %   Y: the actual output of the training data
        
    else
        %% Calculate the training output
        temp = kernel_matrix(temp_inputLayer{s,1}',Kernel_type, kernel_para(layers));
        OutputWeight_final{s,1}=((temp+speye(train_num)/C(layers))\(T'));
        Y{s,1} = OutputWeight_final{s,1}'*temp;                             %   Y: the actual output of the training data
    end
end

%% Testing Phase 
TY = cell(kelm_num,1);
for s = 1:kelm_num
    Omega_test = (zscore(test_x{s,1}'))';    
    Omega_test =  (OutputWeight{s,1}*Omega_test' );
    
    if layers>=3
        for i = 2:(layers-1)
            Omega_test =  (OutputWeight{s,i}*Omega_test);
        end

        Omega_test = kernel_matrix(temp_inputLayer{s,1}',Kernel_type, kernel_para(layers), Omega_test');
        
        TY{s,1}= OutputWeight_final{s,1}'*Omega_test; %   TY: the actual output of the testing data
    else
        Omega_test = kernel_matrix(temp_inputLayer{s,1}',Kernel_type, kernel_para(layers), Omega_test');
        
        TY{s,1}= OutputWeight_final{s,1}'*Omega_test; %   TY: the actual output of the testing data
    end
end


%% Calculate the training accuracy
A=zeros(kelm_num,class_num);    % Store the accuracy of each branch
label_expected_train = zeros(1,train_num);
label_actual_train = zeros(1,train_num);
for s = 1:kelm_num
    MissClassificationRate_Training=0;
    for k = 1 : train_num
        [~, label_index_expected]=max(T(:,k));
        label_expected_train(1,k) = label_index_expected; 
        Y_temp = Y{s,1};
        [~, label_index_actual]=max(Y_temp(:,k));
        label_actual_train(1,k) = label_index_actual;
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    [~, ~, ~,~, ua,~]=confusion(label_expected_train,label_actual_train);
    A(s,:) = ua';
    TrainingAccuracy=1-MissClassificationRate_Training/train_num
end

U = A./sum(A,1);    %平均精度
%% Testing Whole output
wholeY = cell(kelm_num,1);
for s = 1:kelm_num
    Omega_whole = (zscore(whole_x{s,1}'))';   
    Omega_whole =  (OutputWeight{s,1}*Omega_whole' );
    
    if layers>=3
        for i = 2:(layers-1)
            Omega_whole =  (OutputWeight{s,i}*Omega_whole);
        end

        Omega_whole = kernel_matrix(temp_inputLayer{s,1}',Kernel_type, kernel_para(layers), Omega_whole');
        
        wholeY{s,1}= OutputWeight_final{s,1}'*Omega_whole; %   TY: the actual output of the testing data
    else
        Omega_whole = kernel_matrix(temp_inputLayer{s,1}',Kernel_type, kernel_para(layers), Omega_whole');
        
        wholeY{s,1}= OutputWeight_final{s,1}'*Omega_whole; %   TY: the actual output of the testing data
    end
end
num=[kelm_num, test_num];
end





      
