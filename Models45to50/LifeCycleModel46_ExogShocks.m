function [z_grid,pi_z]=LifeCycleModel46_ExogShocks(rho_z,sigma_epsilon_z,n_z)

% First, the AR(1) process z
farmertodaoptions.verbose=0; % turn of warning messages when don't hit moments (as they are really annoying when estimating)
[z_grid,pi_z]=discretizeAR1_FarmerToda(0,rho_z,sigma_epsilon_z,n_z,farmertodaoptions);
z_grid=exp(z_grid); % Take exponential of the grid
[mean_z,~,~,~]=MarkovChainMoments(z_grid,pi_z); % Calculate the mean of the grid so as can normalise it
z_grid=z_grid./mean_z; % Normalise the grid on z (so that the mean of z is exactly 1)

end