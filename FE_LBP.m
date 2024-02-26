function Feature_L = FE_LBP(data,data_gt)

% init
[nr nc nb] = size(data);
test = double(data_gt);
[rows_gt,cols_gt] = size(data_gt);
    z = data;
    Data0 = z./max(z(:));           
    [m n d] = size(Data0);
    map = data_gt;

    %X = reshape(Data0, m*n, d);
    X = Data0;
    Psi = PCA_Train(X', 1);
    X = X*Psi;
    Data = reshape(X, rows_gt, cols_gt, size(Psi,2));

    % LBP feature extraction
    r = 1;  nr = 8;%r is radius;nr is num_point of every cell
    mapping = getmapping(nr,'u2'); 
    Feature_L = LBP_feature_global(Data, r, nr, mapping, 10, map);
    Feature_L = reshape(Feature_L, rows_gt*cols_gt, []);
    %Feature_L = Feature_L(:,1:30);

save Feature_L.mat Feature_L
end

