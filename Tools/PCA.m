function dataPCA = PCA(data,PC)
Sigma = cov(data);
[V,D] = eig(Sigma); % Compute the eigenvalues/eigenvectors
[D,~] = sort(diag(D),'descend');
[~,ind] = sort(cumsum(D)/sum(D),'descend');

V = V(:,ind);

Dp = data*V(:,1:PC);% Projet the data on the d first eigenvector
dataPCA = Dp;
end

