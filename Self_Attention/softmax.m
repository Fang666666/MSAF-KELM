function [Z_softmax] = softmax(input_matrix)
Z = input_matrix - max(input_matrix,[],2);
Z_exp = exp(Z);
Z_softmax = Z_exp ./ sum(Z_exp,2);
Z_softmax = Z_softmax;
end

