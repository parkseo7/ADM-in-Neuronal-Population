% Script to generate a numerical simulation of the neuronal population
% model over time. Uses imported databases from the UMCD website.

% Clear
clear;

% Add to function path
addpath('matrices');
addpath('fcns');

% PARAMETER .CSV FILE
paramName = 'paramSet1.csv';

% Set up directory to export (check if it exists)
cwd = pwd ;

% Import parameters from a .csv file (excel spreadsheet)
dir_params = fullfile(cwd, 'parameters', paramName);
paramCells = readcell(dir_params); % Make sure to clear contents in empty excel cells
params = cell2struct(paramCells(2:end,2), paramCells(2:end,1), 1);

% Import connection matrices from 'matrices' folder
importName = params.importName;
dir_W = fullfile(cwd, 'matrices', importName, 'icbm_fiber_mat.txt');
dir_pos = fullfile(cwd, 'matrices', importName, 'fs_region_centers_68_sort.txt');

W_raw = dlmread(dir_W); % Symmetric matrix
pos = dlmread(dir_pos); % Positions of nodes

if (params.normalizeW == "TRUE")
    W = W_raw / max(W_raw(:)); % Normalize matrix
else
    W = W_raw;
end

% Add parameters based on data set:
N = size(pos, 1);
r0 = params.r00 * ones(N,1);

% Compute initial delays
dist = pdist2(pos, pos); % Distances between nodes (in mm)
v0 = unifrnd(params.vLow, params.vHigh, N,N);
tau0 = dist ./ v0;

% Convert units (if in seconds)
if (params.inSeconds == "TRUE")
    params.kappa = params.kappa / 1000;
    params.C1 = params.C1 / 1000;
    params.C2 = params.C2 * 1000;
    dist = dist / 1000;
    tau0 = tau0 / 1000;
    r0 = r0 * 1000;
end

% Update parameters
params.N = N;
params.r0 = r0;
params.W = W;
params.W_raw = W_raw;
params.dist = dist;
params.v0 = v0;
params.tau0 = tau0;

% IMPLEMENT LEARNING:
initime = cputime;
output = ADMLearn(params);
fintime = cputime;
output.comp_time = cputime;

% EXPORT
dir_folder = fullfile(cwd, 'arrays', 'ICBM_Matlab', params.exportName) ;
if ~exist(dir_folder, 'dir')
   mkdir(dir_folder)
end
dir_results = fullfile(dir_folder, 'results');
dir_params = fullfile(dir_folder, 'params');

save(dir_params, '-struct', 'params');
save(dir_results, '-struct', 'output');