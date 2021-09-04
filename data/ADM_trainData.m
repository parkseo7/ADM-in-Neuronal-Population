% Script to generate a numerical simulation of the neuronal population
% model over time.

% Clear
clear;

% Add to function path
addpath('matrices');
addpath('fcns');

% Import connection matrices from 'matrices' folder
importName = 'train';
cwd = pwd;
dir_W = fullfile(cwd, 'matrices', importName, 'W_train1.txt');
dir_pos = fullfile(cwd, 'matrices', importName, 'pos_train1.txt');

W = dlmread(dir_W); % Symmetric matrix
% W = W / max(W(:)); % Normalize matrix
pos = dlmread(dir_pos); % Positions of nodes


% In ms or s?
is_sec = true;

% Parameters (custom)
r00 = 0.1; % Baseline uniform rate (in mHz)
vRange = [0.5, 2.0]; % Initial velocities
numIters = 100; % Number of iterations
kappa = 2.5; % Std of coincidence (in ms)
eta = 0.001; % Base learning rate

% Set up other parameters
N = size(pos, 1);
r0 = r00 * ones(N,1);

% Compute initial delays
dist = pdist2(pos, pos); % Distances between nodes (in mm)
% v0 = unifrnd(vRange(1), vRange(2), N,N);
v0 = 1.1 * ones(N,N);
tau0 = dist ./ v0;

% Convert units (if in seconds)
if (is_sec)
    kappa = kappa / 1000;
    dist = dist / 1000;
    tau0 = tau0 / 1000;
    r0 = r0 * 1000;
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
    'eta', eta, ...
    'numIters', numIters ...
    );

% Adagrad parameters
params.epsilon = 1e-8;
params.avg_fac = 0.9;

% Sample size for output arrays (as a percentage of nonzero connections)
params.sam_size = 0.10;

% IMPLEMENT LEARNING:
initime = cputime;
output = ADMLearn(params, 1);
fintime = cputime;

comp_time = fintime - initime;

% Objective function
rates = output.rE;
obj_arr = sum(rates.^2, 1)/2;
eigs = output.eigs;

figure
plot(1:numIters, obj_arr/1000);