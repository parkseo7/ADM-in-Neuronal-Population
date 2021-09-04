function [rE, eigs] = computeRatesEigs(W, gamma, r0, N)
% Compute the rates and stability eigenvalues at the current iteration

A = W .* gamma / N - eye(N);
eigs = eig(A);
rE = -A \ r0;

end
