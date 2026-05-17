%% Life-Cycle Model 30: Linear Interpolation of aprime (Grid Interpolation Layer)
% Solves the same model as Life-Cycle Model 9, twice. First the usual way, then with vfoptions.gridinterplayer=1, which adds a 'grid interpolation layer'
% that lets aprime take (vfoptions.ngridinterp=20) values between the grid points on a_grid (using linear interpolation). For any given n_a the grid
% interpolation layer is marginally slower but more accurate (in particular, it allows aprime to take values that are not on a_grid).
%
% Unlike with divide-and-conquer (Life-Cycle Model 29), turning on the grid interpolation layer changes the answer: both V changes, and even the size
% of Policy changes. Because of this, you have to tell simoptions about the grid interpolation layer settings.
% We then also compute the agent distribution twice and see that StationaryDist differs slightly between the two solutions.
%
% In practice, because grid interpolation layer is more accurate, you can use a smaller n_a than you otherwise would, and so the model is both faster and
% more accurate (and uses less GPU memory).
%
% We can use divide-and-conquer together with grid interpolation layer, and we solve the value fn iteration a third time just to demonstrate.

%% How does VFI Toolkit think about this?
%
% One decision variable: h, labour hours worked
% One endogenous state variable: a, assets (total household savings)
% One stochastic exogenous state variable: z, an AR(1) process (in logs), idiosyncratic shock to labor productivity units
% Age: j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=51; % Endogenous labour choice (fraction of time worked)
n_a=201; % Endogenous asset holdings
n_z=21; % Exogenous labor productivity units shock
N_j=Params.J; % Number of periods in finite horizon

%% Parameters

% Discount rate
Params.beta = 0.96;
% Preferences
Params.sigma = 2; % Coeff of relative risk aversion (curvature of consumption)
Params.eta = 1.5; % Curvature of leisure (This will end up being 1/Frisch elasticity)
Params.psi = 10; % Weight on leisure

% Prices
Params.w=1; % Wage
Params.r=0.05; % Interest rate (0.05 is 5%)

% Demographics
Params.agej=1:1:Params.J; % Is a vector of all the agej: 1,2,3,...,J
Params.Jr=46;

% Pensions
Params.pension=0.3;

% Age-dependent labor productivity units
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
% Exogenous shock process: AR1 on labor productivity units
Params.rho_z=0.9;
Params.sigma_epsilon_z=0.03;

% Conditional survival probabilities: sj is the probability of surviving to be age j+1, given alive at age j
% Most countries have calculations of these (as they are used by the government departments that oversee pensions)
% In fact I will here get data on the conditional death probabilities, and then survival is just 1-death.
% Here I just use them for the US, taken from "National Vital Statistics Report, volume 58, number 10, March 2010."
% I took them from first column (qx) of Table 1 (Total Population)
% Conditional death probabilities
Params.dj=[0.006879, 0.000463, 0.000307, 0.000220, 0.000184, 0.000172, 0.000160, 0.000149, 0.000133, 0.000114, 0.000100, 0.000105, 0.000143, 0.000221, 0.000329, 0.000449, 0.000563, 0.000667, 0.000753, 0.000823,...
    0.000894, 0.000962, 0.001005, 0.001016, 0.001003, 0.000983, 0.000967, 0.000960, 0.000970, 0.000994, 0.001027, 0.001065, 0.001115, 0.001154, 0.001209, 0.001271, 0.001351, 0.001460, 0.001603, 0.001769, 0.001943, 0.002120, 0.002311, 0.002520, 0.002747, 0.002989, 0.003242, 0.003512, 0.003803, 0.004118, 0.004464, 0.004837, 0.005217, 0.005591, 0.005963, 0.006346, 0.006768, 0.007261, 0.007866, 0.008596, 0.009473, 0.010450, 0.011456, 0.012407, 0.013320, 0.014299, 0.015323,...
    0.016558, 0.018029, 0.019723, 0.021607, 0.023723, 0.026143, 0.028892, 0.031988, 0.035476, 0.039238, 0.043382, 0.047941, 0.052953, 0.058457, 0.064494,...
    0.071107, 0.078342, 0.086244, 0.094861, 0.104242, 0.114432, 0.125479, 0.137427, 0.150317, 0.164187, 0.179066, 0.194979, 0.211941, 0.229957, 0.249020, 0.269112, 0.290198, 0.312231, 1.000000];
% dj covers Ages 0 to 100
Params.sj=1-Params.dj(21:101); % Conditional survival probabilities
Params.sj(end)=0; % In the present model the last period (j=J) value of sj is actually irrelevant

% Warm glow of bequest
Params.wg1=0.3; % (relative) importance of bequests
Params.wg2=3; % degree to which bequests are a luxury good (>=1; =1 would be a normal good)
Params.wg3=Params.sigma; % By using the same curvature as the utility of consumption it makes it much easier to guess appropriate parameter values for the warm glow


%% Grids
% The ^3 means that there are more points near 0 and near 10. We know from theory that the value function will be more 'curved' near zero assets,
% and putting more points near curvature (where the derivative changes the most) increases accuracy of results.
a_grid=10*(linspace(0,1,n_a).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.

% First, the AR(1) process z
[z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_epsilon_z,n_z);
z_grid=exp(z_grid); % Take exponential of the grid
[mean_z,~,~,~]=MarkovChainMoments(z_grid,pi_z); % Calculate the mean of the grid so as can normalise it
z_grid=z_grid./mean_z; % Normalise the grid on z (so that the mean of z is exactly 1)

% Grid for labour choice
h_grid=linspace(0,1,n_d)'; % Notice that it is imposing the 0<=h<=1 condition implicitly
% Switch into toolkit notation
d_grid=h_grid;

%% Now, create the return function
DiscountFactorParamNames={'beta','sj'};

% Notice we still use 'LifeCycleModel8_ReturnFn'
ReturnFn=@(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,wg1,wg2,wg3,beta,sj)...
    LifeCycleModel8_ReturnFn(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,wg1,wg2,wg3,beta,sj);

%% Solve the value function iteration problem, the usual way
disp('Solve for Value fn and Policy fn using ValueFnIter command')
vfoptions=struct(); % Just using the defaults.
tic;
[V1, Policy1]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
vftime1=toc

%% Solve the value function iteration problem, with the grid interpolation layer
% The grid interpolation layer puts vfoptions.ngridinterp evenly-spaced points
% between each pair of consecutive points on a_grid, and allows aprime to take
% values on this finer 'interpolation' grid (the value function at those
% in-between points is computed by linear interpolation).
vfoptions.gridinterplayer=1; % turn on the grid interpolation layer
vfoptions.ngridinterp=20; % 20 evenly-spaced points between each pair of consecutive a_grid points
tic;
[V2, Policy2]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
vftime2=toc

%% Compare the two solutions
% Unlike with divide-and-conquer, turning on the grid interpolation layer changes the answer: V differs (the second solution is more accurate, since
% aprime is allowed to take values not on a_grid), and even the size of Policy changes (there is now an extra entry storing aprime's position on the
% interpolation layer between the two relevant a_grid points).
size(Policy1) % size [length(n_d)+length(n_a),n_a,n_z,N_j]
size(Policy2) % size [length(n_d)+length(n_a)+1,n_a,n_z,N_j]; one more entry, for the interpolation layer position
max(abs(V1(:)-V2(:))) % nonzero: V differs because aprime can now take values that are not on a_grid

% Note: for any given n_a, the grid interpolation layer will be marginally slower but more accurate than the standard solution.

%% Solve the value function iteration problem, with both divide-and-conquer and the grid interpolation layer
% Just to emphasize that the two options can be used together.
vfoptions.divideandconquer=1; % turn on divide-and-conquer
tic;
[V3, Policy3]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
vftime3=toc

%% Compute the agent distribution, twice (without and with grid interpolation layer)

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,floor((n_z+1)/2))=1; % All agents start with zero assets, and the median shock

%% Compute the 'stationary distribution' of households
% Start with a mass of one at initial age, use the conditional survival
% probabilities sj to calculate the mass of those who survive to next
% period, repeat. Once done for all ages, normalize to one
Params.mewj=ones(1,Params.J); % Marginal distribution of households over age
for jj=2:length(Params.mewj)
    Params.mewj(jj)=Params.sj(jj-1)*Params.mewj(jj-1);
end
Params.mewj=Params.mewj./sum(Params.mewj); % Normalize to one
AgeWeightsParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age

% First, without the grid interpolation layer
simoptions=struct(); % Just using the defaults.
StationaryDist1=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy1,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);

% Now, with the grid interpolation layer
% Because the grid interpolation layer changes the size and interpretation of
% Policy, we have to tell the stationary distribution command about it too, by
% setting simoptions.gridinterplayer and simoptions.ngridinterp (to the same
% values we used in vfoptions).
simoptions.gridinterplayer=1; % turn on the grid interpolation layer
simoptions.ngridinterp=20;
StationaryDist2=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy2,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);

% The two stationary distributions differ. The difference is small but not zero.
max(abs(StationaryDist1(:)-StationaryDist2(:)))

%% Accuracy: as n_a grows, the two solutions converge
% With n_a large enough the values of aprime that the grid interpolation layer can pick will be very close to a_grid points anyway, so the two solutions
% become essentially identical. Try setting n_a to a large value (e.g., 1001 or 2001) at the top of this file and rerun: you will see that the differences
% in V and StationaryDist between the two solutions get smaller and smaller as n_a gets larger.
