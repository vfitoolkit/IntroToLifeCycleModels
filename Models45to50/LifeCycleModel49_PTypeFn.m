function Params=LifeCycleModel49_PTypeFn(Params)
% When setting up the Permanent Types as a function just needs to take the
% parameters structure as an input, and return the parameters structure as
% an output.

% For Life-Cycle Model 49:
% We use logalpha_mean and logalpha_stddev to parametrize the distribution of the permanent types
% The distribution consists of alpha_i and alphadist (values and masses, respectively)
% So this function needs to use the values of logalpha_mean and
% logalpha_stddev to update the values of alpha_i and alphadist.

[alpha_grid,pi_alpha]=discretizeAR1_FarmerToda(Params.logalpha_mean,0,Params.logalpha_stddev,Params.N_i); % discretize normal distribution
pi_alpha=pi_alpha(1,:); % iid, just use first row

alpha_grid=exp(alpha_grid);
alpha_grid=alpha_grid*exp(Params.logalpha_mean)/sum(alpha_grid.*pi_alpha'); % normalize to mean exp(Params.logmean_stddev)

% alpha_i and alphadist are both stored in PTypeParams, and so will VFI
% Toolkit will put them into Params and we can use them like any other parameter
Params.alpha_i=alpha_grid;
Params.alphadist=pi_alpha; % Must sum to one


end