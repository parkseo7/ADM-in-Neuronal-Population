function gamma = coincidenceFactor(W, tau, kappa, N)
% Compute the coincidence factor matrix at the current iteration

gamma = zeros(N);
for i = 1:N
    v_i = tau(i,:);
    W_sum = sum(W(i,:));
    W_sum = W_sum + (W_sum == 0);
    W_i = W(i,:) / W_sum;
    diff_i = bsxfun(@minus, v_i, v_i.');
    gamma_i = exp(-0.5 * diff_i.^2 / kappa^2) * W_i.';
    gamma(i,:) = gamma_i;
    
end

end
