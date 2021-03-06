% Script to generate a numerical simulation of the neuronal population
% model over time. Uses imported databases from the UMCD website.

% Clear
clear;

% Add to function path
addpath('matrices');
addpath('fcns');

% Set up directory to export (check if it exists)
exportName = 'trial8' ;
cwd = pwd ;
dir_folder = fullfile(cwd, 'arrays', 'ICBM_Matlab', exportName) ;

if ~exist(dir_folder, 'dir')
   mkdir(dir_folder)
end

% Import connection matrices from 'matrices' folder
importName = 'ICBM';
dir_W = fullfile(cwd, 'matrices', importName, 'icbm_fiber_mat.txt');

dir_pos = fullfile(cwd, 'matrices', importName, 'fs_region_centers_68_sort.txt');

W_raw = dlmread(dir_W); % Symmetric matrix
W = W_raw / max(W_raw(:)); % Normalize matrix
pos = dlmread(dir_pos); % Positions of nodes


% In ms or s?
is_sec = true;

% Parameters (custom)
r00 = 0.1; % Baseline uniform rate (in mHz)
vRange = [0.1, 0.5]; % Initial velocities
numIters = 2000; % Number of iterations
kappa = 30; % Std of coincidence (in ms)
eta = 1.2 * 1e-3; % 1e-3; % Base learning rate
% Set up other parameters
N = size(pos, 1);
r0 = r00 * ones(N,1);

alpha1 = 0.2/N; % Cost function coefficient
alpha2 = 5.0 / 1000 / N; % Cost function exponential coefficient

% Compute initial delays
dist = pdist2(pos, pos); % Distances between nodes (in mm)
v0 = unifrnd(vRange(1), vRange(2), N,N);
tau0 = dist ./ v0;

% Convert units (if in seconds)
if (is_sec)
    kappa = kappa / 1000;
    dist = dist / 1000;
    tau0 = tau0 / 1000;
    r0 = r0 * 1000;
    alpha2 = alpha2 * 1000;
    alpha1 = alpha1 / 1000;
end

% Store parameters in structure
params = struct( ...
    'N', N, ...
    'W', W, ...
    'r0', r0, ...
    'dist', dist, ...
    'v0', v0, ...
    'tau0', tau0, ...
    'kappa', kappa, ...
    'alpha1', alpha1, ...
    'alpha2', alpha2, ...
    'eta', eta, ...
    'numIters', numIters ...
    );

% Gradient mode
gradientMode = 2;

% Adagrad parameters
params.epsilon = 1e-8;
params.avg_fac = 0.90;

% Sample size for output arrays (as a percentage of nonzero connections)
params.sam_size = 0.10;

% IMPLEMENT LEARNING:
initime = cputime;
output = ADMLearn(params, gradientMode); % 1 for Adadelta
fintime = cputime;

comp_time = fintime - initime;

% PROCESS OUTPUT ARRAYS
rates = output.rE;
eigs = output.eigs;
indsSam = output.inds;
tau = output.tau;
vel = output.vel;
grad = output.grad;
slope = output.slope;
gamma = output.gamma;
objective = sum(rates.^2, 1)/2 - alpha1 * sum(exp(alpha2 * tau));
mode = output.mode;

% Maximum rates
[ratesMax,~] = computeRatesEigs(W, ones(N,1), r0, N);
objMax = sum(ratesMax.^2) / 2;

% Initial and final histograms
gamma0 = output.gamma0;
gammaf = output.gammaf;
tauf = output.tauf;
velf = output.velf;

% EXPORT
dir_results = fullfile(dir_folder, 'results');
dir_params = fullfile(dir_folder, 'params');

% Parameters
save(dir_params, ...
    'N', 'W', 'W_raw', 'r0', 'dist', 'v0', ...
    'tau0', 'kappa', 'eta', 'numIters', ...
    'vRange', 'alpha1', 'alpha2' ...
    );

% Results
save(dir_results, ...
    'rates', ...
    'eigs', ...
    'indsSam', ...
    'tau', ...
    'vel', ...
    'grad', ...
    'slope', ...
    'gamma', ...
    'objective', ...
    'objMax', ...
    'gamma0', ...
    'gammaf', ...
    'tauf', ...
    'velf', ...
    'comp_time', ...
    'mode' ...
    );

    







