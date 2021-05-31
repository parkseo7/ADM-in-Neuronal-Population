% Script to generate a numerical simulation of the neuronal population
% model over time.

% Clear
clear;

% Add to function path
addpath('matrices');
addpath('fcns');

% Set up directory (check if it exists)
foldername = 'trial1' ;
cwd = pwd ;
dir_folder = fullfile(cwd, 'arrays', foldername) ;

if ~exist(dir_folder, 'dir')
   mkdir(dir_folder)
end

filename = 'results.mat'; % .csv file name
dir_file = fullfile(dir_folder, filename); % Export directory

% Import connection matrices from 'matrices' folder
dir_W = fullfile(cwd, 'matrices', 'fs85_connectmat.txt');
dir_pos = fullfile(cwd, 'matrices', 'fs85_centers.txt');

W = dlmread(dir_W); % Symmetric matrix
pos = dlmread(dir_pos); % Positions of nodes

% Parameters
N = size(pos, 1);
r0 = 1.0 * ones(N,1); % Initial activation rates
v0 = 5.0; % Initial velocity
kappa = 0.0005;

t0 = 0; % Initial time (in seconds)
tf = 2500; % Final time (in seconds)

% Compute initial delays
dist = pdist2(pos, pos);
tau0 = dist / v0;

% Iterative learning
eta = 0.2;
num_iter = 1500;
eigs_iter = zeros(N, num_iter);
rE_iter = zeros(N, num_iter);
tau = tau0;

% Wait bar
f = waitbar(0,'Starting trials...') ;

for k=1:num_iter+1
    
    % Waitbar
    waittext = ['Iteration: ' num2str(k)] ;
    waitprog = k / num_iter ;
    waitbar(waitprog, f, waittext) ;
    
    
    gamma = zeros(N);
    tau_diffs = zeros(N);
    getGamma = @(i,j) exp(-kappa/2 * sum((tau(i,j) - tau(:,j)).^2));
    getDiffs = @(i,j) sum(tau(i,j) - tau(:,j));
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
    tau = tau - eta*kappa*bsxfun(@times, rE', rE).*W.*gamma*tau_diffs/N; 
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
    'tau', 'dist', 'eigs_iter', 'rE_iter', 't', 'y');