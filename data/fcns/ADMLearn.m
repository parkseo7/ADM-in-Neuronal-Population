function output = ADMLearn(params)
% Implement the learning rule on the conduction velocities, given the data.
% Returns arrays of plottable information

% PARAMETERS
N = params.N;
W = params.W;
r0 = params.r0;
dist = params.dist;
v0 = params.v0;
tau0 = params.tau0;
kappa = params.kappa;
eta = params.eta;
numIters = params.numIters;

alpha1 = params.C1 / N;
alpha2 = params.C2 / N^2;

mode = params.gradientMode;
epsilon = params.epsilon;
avg_fac = params.avgFac;
sam_size = params.samSize;

% Sample size for output arrays
W_inds = find(W ~= 0);
M = int16(sam_size * numel(W_inds)); % Sampling size as a portion of valid connections
inds = datasample(W_inds, M, 'replace', false);
inds = sort(inds);

% OUTPUT ARRAYS
rE_arr = zeros(N, numIters);
eigs_arr = zeros(N, numIters);
tau_arr = zeros(M, numIters);
vel_arr = zeros(M, numIters);
grad_arr = zeros(M, numIters);
slope_arr = zeros(M, numIters);
gamma_arr = zeros(M, numIters);
obj_arr = zeros(numIters, 1);


% INITIALIZE
tau = tau0;
vel = v0;
gamma0 = coincidenceFactor(W, tau0, kappa, N);

ada_fac = zeros(N);
slope_prev = zeros(N);
BV = zeros(N);

% Wait bar
f = waitbar(0,'Starting trials...') ;

% MAIN LOOP
for k=1:numIters
    
    % Waitbar
    waittext = ['Iteration: ' num2str(k) ' out of ' num2str(numIters)] ;
    waitprog = k / numIters ;
    waitbar(waitprog, f, waittext) ;
    
    % SOLVE FOR SOLUTIONS:
    gamma = coincidenceFactor(W, tau, kappa, N);
    [rE, eigs] = computeRatesEigs(W, gamma, r0, N);
    rE = reshape(rE, N,1);
    eigs = reshape(eigs, N,1);
    
    % SOLVE FOR GRADIENT:
    gradL_nonzero = zeros(numel(W_inds), 1);
    parfor l = 1:numel(W_inds)
        gradL_nonzero(l) = derivObjectiveTau(W, tau, kappa, gamma, rE, N, W_inds(l));
    end
    
    % Reshape gradL:
    gradL = zeros(N^2, 1);
    gradL(W_inds) = gradL_nonzero;
    
    
    gradL = reshape(gradL, N,N);
    
    % Safe code:
    % gradL = zeros(N);
    % parfor m=1:N^2
        % gradL(m) = derivObjectiveTau(W, tau, kappa, gamma, rE, N, m);
    % end
    
    % Cost function
    cost = alpha1 * alpha2 * exp(alpha2 * tau);
    gradL_vel = (gradL - cost) .* (-dist ./ (vel.^2));
    
    % Implment gradient descent. May need to impose decay of eta over time.
    if (mode == "adagrad") % Adagrad
        ada_fac = avg_fac * ada_fac + (1 - avg_fac) * gradL_vel.^2;
        slope = gradL_vel ./ (sqrt(ada_fac + epsilon));
        
    elseif (mode == "adagradBV") % Adagrad with bounded variation
        ada_fac = avg_fac * ada_fac + (1 - avg_fac) * gradL_vel.^2;
        slope = gradL_vel ./ (sqrt(ada_fac + epsilon));
        
        % If slope exceeds BV, reduce it
        slopeMult = ones(N) - 0.5*(abs(slope - slope_prev) > 0.75*BV);
        slope = slopeMult .* slope;
        
        % Bounded variation (with would-be slope)
        BV = ((k-1)*BV + abs(slope_prev - slope)) / k;
        slope_prev = slope;
        
    else
        slope = gradL_vel / (1 + norm(gradL_vel));
    end
    
    vel = vel + eta * slope;
    tau = dist ./ vel;
    % STORE INTO ARRAYS:
    rE_arr(:,k) = rE;
    eigs_arr(:,k) = eigs;
    tau_arr(:,k) = tau(inds);
    vel_arr(:,k) = vel(inds);
    grad_arr(:,k) = gradL_vel(inds);
    slope_arr(:,k) = slope(inds);
    gamma_arr(:,k) = gamma(inds);
    obj_arr(k) = sum(rE.^2)/2 - alpha1 * sum(exp(alpha2 * tau), 'all');
end

close(f);

% Process arrays
[ratesMax,~] = computeRatesEigs(W, ones(N,1), r0, N);
objMax = sum(ratesMax.^2) / 2;

% Final values after learning
tauf = tau;
velf = vel;
gammaf = gamma;

% CONFIGURE OUTPUT:
output = struct( ...
    'rE', rE_arr, ...
    'eigs', eigs_arr, ...
    'inds', inds, ...
    'tau', tau_arr, ...
    'vel', vel_arr, ...
    'grad', grad_arr, ...
    'slope', slope_arr, ...
    'gamma', gamma_arr, ...
    'gamma0', gamma0, ...
    'gammaf', gammaf, ...
    'tauf', tauf, ...
    'velf', velf, ...
    'obj_arr', obj_arr, ...
    'objMax', objMax, ...
    'mode', mode, ...
    'alpha1', alpha1, ...
    'alpha2', alpha2 ...
    );

end


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


function [rE, eigs] = computeRatesEigs(W, gamma, r0, N)
% Compute the rates and stability eigenvalues at the current iteration

A = W .* gamma / N - eye(N);
eigs = eig(A);
rE = -A \ r0;

end


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