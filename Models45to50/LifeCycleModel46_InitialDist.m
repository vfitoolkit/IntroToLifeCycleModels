function jequaloneDist=LifeCycleModel46_InitialDist(a_grid,z_grid,n_a,n_z,initassets_mean,initz_mean,ah2021p1,ah2021p2,ah2021p3)

% Trying to simply parametrize a covariance matrix is not a good approach to estimation
% So instead, we follow Acharkov & Hansen (2021)
% http://discourse.vfitoolkit.com/t/parametrize-a-covariance-matrix-using-archakov-hansen-2021/252

% Input parameters are the AH2021 parametrization. So create the covariance matrix from this.

% Trivially first step, split the vector into the part relating to std
% deviation and the part relating to the correlation matrix
StdDevVector=[ah2021p1,ah2021p2];
AHcorrvector=ah2021p3;

% We use AH2021 to turn this later vector into the correlation matrix
tol_value=10^(-9); % AH2021 require a tolerance (between 10^(-4) and 10^(-14))
[CorrMatrix2, iter_number ] = GFT_inverse_mapping(AHcorrvector, tol_value);
% iter_number is the number of iterations it took to converge (to the tolerance declared in tol_value)
% Note: AH2021 provide GFT_inverse_mapping() in their Online Appendix (for multiple
% programming languages). The version used here is a lightly modified/cleaned version of theirs.

% And then use Matlab corr2cov() to convert the vector of std deviations
% and correlation matrix into the covariance matrix
InitialDistCovarMatrix = corr2cov(StdDevVector,CorrMatrix2);


% So we need to put this joint-normal distribution onto our asset grid
jequaloneDist=MVNormal_ProbabilitiesOnGrid([a_grid; z_grid],[initassets_mean; initz_mean],InitialDistCovarMatrix,[n_a,n_z]); % note: first point in a_grid is zero, so have to add something tiny before taking log

end