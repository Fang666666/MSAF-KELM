function [Z_ReLU] = ReLU(input_matrix)
Z = input_matrix;
Z_ReLU = max(0,Z);
end

