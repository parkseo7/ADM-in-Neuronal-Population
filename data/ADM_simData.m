% Script to generate a numerical simulation of the neuronal population
% model over time.

% Clear
clear;

% Add to function path
addpath('matrices');
addpath('fcns');

% Set up directory (check if it exists)
foldername = 'ICBM' ;
cwd = pwd ;
dir_folder = fullfile(cwd, 'arrays', foldername) ;

if ~exist(dir_folder, 'dir')
   mkdir(dir_folder)
end

filename = 'results2.mat'; % .csv file name
dir_file = fullfile(dir_folder, filename); % Export directory

% Import connection matrices from 'matrices' folder
importname = 'ICBM';
dir_W = fullfile(cwd, 'matrices', importname, 'icbm_fiber_mat.txt');
dir_pos = fullfile(cwd, 'matrices', importname, 'fs_region_centers_68_sort.txt');

W = dlmread(dir_W); % Symmetric matrix
% W = (W > 0.0002) .* W ; % Remove all negligible connections
W = W / max(W(:)); % Normalize matrix
pos = dlmread(dir_pos); % Positions of nodes

% Parameters
N = size(pos, 1);
r0 = 1.0 * ones(N,1); % Initial activation rates
v0 = 3.0; % Initial velocity
kappa = 0.0005;

t0 = 0; % Initial time (in seconds)
tf = 2500; % Final time (in seconds)

% Compute initial delays
dist = pdist2(pos, pos);
tau0 = dist / v0;

% Iterative learning
eta = 100 / kappa;
num_iter = 6000;
eigs_iter = zeros(N, num_iter);
rE_iter = zeros(N, num_iter);
tau = tau0;

% Initial gamma check (delete after)
gamma0 = zeros(N);
getGamma0 = @(i,j) exp(-kappa/2 * sum((W(i,:) > 0) .* (tau0(i,j) - tau0(i,:)).^2));
for j1=1:N
    for j2=1:N
        gamma0(j1,j2) = getGamma0(j1,j2);
    end
end

        
% Wait bar
f = waitbar(0,'Starting trials...') ;

for k=1:num_iter+1
    
    % Waitbar
    waittext = ['Iteration: ' num2str(k)] ;
    waitprog = k / num_iter ;
    waitbar(waitprog, f, waittext) ;
    
    
    gamma = zeros(N);
    tau_diffs = zeros(N);
    getGamma = @(i,j) exp(-kappa/2 * sum((W(i,:) > 0) .* (tau(i,j) - tau(i,:)).^2)); % Use W?
    getDiffs = @(i,j) sum((W(i,:) > 0) .* (tau(i,j) - tau(i,:))); % Weighted here as well?
    for k1=1:N
        for k2=1:N
            gamma(k1,k2) = getGamma(k1,k2);
            tau_diffs(k1,k2) = getDiffs(k1,k2);
        end
    end
    
    % Obtain eigenvalues + equilibria
    A = W .* gamma / N - eye(N);
    eigs_iter(:,k) = eig(A);
    rE = -A \ r0;
    rE_iter(:,k) = rE;
    
    if k == num_iter
        break;
    end
    
    % Adjust delays with gradient descent:
    tau = tau - eta*kappa*bsxfun(@times, rE', rE).*W.*gamma.*tau_diffs/N; 
end

close(f);

% Solver options
options = odeset();

% Run simulation
sol = solveADMModel(t0,tf,W,gamma,r0,options);
t = sol.x;
y = sol.y;

% Export arrays and parameters
save(dir_file, 'N', 'r0', 'v0', 'kappa', 'W', 'pos', 't0', 'tf', 'eta', 'tau0', ...
    'tau', 'dist', 'eigs_iter', 'rE_iter', 'gamma', 't', 'y');