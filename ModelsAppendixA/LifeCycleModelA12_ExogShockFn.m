function [z_grid,pi_z]=LifeCycleModelA12_ExogShockFn(rho_z,sigma_epsilon_z,n_z)
% Note: contents are essentially just copy-paste of how we created z_grid and pi_z in Life-Cycle Model 11.

[z_grid,pi_z]=discretizeAR1_FarmerToda(0,rho_z,sigma_epsilon_z,n_z);
z_grid=exp(z_grid); % Take exponential of the grid
[mean_z,~,~,~]=MarkovChainMoments(z_grid,pi_z); % Calculate the mean of the grid so as can normalise it
z_grid=z_grid./mean_z; % Normalise the grid on z (so that the mean of z is 1)

end