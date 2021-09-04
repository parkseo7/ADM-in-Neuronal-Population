function derivLearning = derivObjectiveTau(W, tau, kappa, gamma, rates, N, ind)
% Compute the derivative of the objective function at index ind = (i,j) as
% an integer, with respect to tau_ij. 

% Convert ind to row, column
[i,j] = ind2sub([N,N], ind);

tau_i = tau(i,:);
W_i = W(i,:);
W_sum = sum(W_i);
W_sum = W_sum + (W_sum == 0);
W_norm_i = W(i,:) / W_sum;

% Set up a linear system
A = eye(N) - W .* gamma / N;
b = zeros(N,1);

diffsTau = tau_i - tau(i,j);
derivGauss = diffsTau .* exp(-0.5 * diffsTau.^2 / kappa^2) / kappa^2;
derivGamma = W_norm_i(j) * derivGauss;
derivGamma(j) = sum(W_norm_i .* derivGauss);
W_i = reshape(W_i, N,1);
derivGamma = reshape(derivGamma, N,1);
b(i) = sum(W_i .* derivGamma .* rates) / N;

% Solve for linear system
derivRates = A \ b;
derivLearning = sum(rates .* derivRates);

end
