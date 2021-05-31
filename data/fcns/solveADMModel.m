function sol = solveADMModel(t0,tf,W,gamma,r0, options)

% Outputs a structure with the solution arrays of the neuronal population
% model, using the given parameters.

N = size(W,1);

% Vector field
drdt = @(t,r) -r + (W.*gamma) * r / N + r0;

sol = ode45(drdt, [t0,tf], r0, options);

end
   
