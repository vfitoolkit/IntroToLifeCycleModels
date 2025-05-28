function [e_grid,pi_e]=LifeCycleModelA12_EiidShockFn(sigma_epsilon_e,n_e)
% Note: contents are essentially just copy-paste of how we created e_grid and pi_e in Life-Cycle Model 11.

[e_grid,pi_e]=discretizeAR1_FarmerToda(0,0,sigma_epsilon_e,n_e);
e_grid=exp(e_grid); % Take exponential of the grid
pi_e=pi_e(1,:)'; % Because it is iid, the distribution is just the first row (all rows are identical). We use pi_e as a column vector for VFI Toolkit to handle iid variables.
mean_e=pi_e'*e_grid; % Because it is iid, pi_e is the stationary distribution (you could just use MarkovChainMoments(), I just wanted to demonstate a handy trick)
e_grid=e_grid./mean_e; % Normalise the grid on z (so that the mean of e is 1)

end