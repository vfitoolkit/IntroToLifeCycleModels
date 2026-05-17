%% Life-Cycle Model 44: Two endogenous states (liquid and illiquid assets)
% When using two endogenous states you essentially put
% (...a1prime,a2prime,a1,a2,...)
% in ReturnFn, FnsToEvaluate, etc.
%
% Models with just one asset do not generate a realistic distribution of
% the marginal propensity to consume. Models with an illiquid asset have
% been identified as one one to improve this and have a more realistic
% distribution of the marginal propensity to consume by creating
% 'wealthy-hand-to-mouth' households which have assets and still have a large
% marginal propensity to consume because their assets are illiquid and they
% have almost no liquid assets.

% Two other aspects of the setup worth highlighting:
% n_a=[5,11] % n_a is the number of grid points for each of the two assets
% a_grid=[a1_grid; a2_grid] % a_grid just stacks the column vector for a1_grid and a2_grid on top of each other

% The illiquid asset incurs an adjustment cost if the quantity changes
% Params.adjcost

%% How does VFI Toolkit think about this?
%
% One decision variable: h, labour hours worked
% Two endogenous state variable: a1 liquid assets and a2 illiquid assets
% Two stochastic exogenous state variables: z and e, persistent and transitory shocks to labor efficiency units, respectively
% Age: j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=5; % Endogenous labour choice (fraction of time worked)
n_a=[21,21]; % Endogenous asset holdings
n_z=11; % Exogenous labor productivity units shocks, persistent and transitory
n_e=3;
N_j=Params.J; % Number of periods in finite horizon

%% Parameters

% Discount rate
Params.beta = 0.96;
% Preferences
Params.sigma = 2; % Coeff of relative risk aversion (curvature of consumption)
Params.eta = 1.5; % Curvature of leisure (This will end up being 1/Frisch elasticity)
Params.psi = 10; % Weight on leisure

% Illiquid asset
Params.adjcost=0.1; % Adjustment cost

% Prices
Params.w=1; % Wage
Params.r1=0.03; % Rate of return on liquid assets
Params.r2=0.06; % Rate of return on illiquid assets

% Demographics
Params.agej=1:1:Params.J; % Is a vector of all the agej: 1,2,3,...,J
Params.Jr=46;

% Pensions
Params.pension=0.3;

% Age-dependent labor productivity units
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
% persistent AR(1) process on idiosyncratic labor productivity units
Params.rho_z=0.9;
Params.sigma_epsilon_z=0.02;
% transitory iid normal process on idiosyncratic labor productivity units
Params.sigma_epsilon_e=0.2; % Implicitly, rho_e=0

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
a1_grid=10*(linspace(0,1,n_a(1)).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.
a2_grid=10*(linspace(0,1,n_a(2)).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.
a_grid=[a1_grid; a2_grid]; % stacked column vectors

% First, the AR(1) process z
if Params.rho_z<0.99
    [z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_epsilon_z,n_z);
elseif Params.rho_z>=0.99 % Rouwenhorst performs better than Farmer-Toda when the autocorrelation is very high
    [z_grid,pi_z]=discretizeAR1_Rouwenhorst(0,Params.rho_z,Params.sigma_epsilon_z,n_z);
end
z_grid=exp(z_grid); % Take exponential of the grid
[mean_z,~,~,~]=MarkovChainMoments(z_grid,pi_z); % Calculate the mean of the grid so as can normalise it
z_grid=z_grid./mean_z; % Normalise the grid on z (so that the mean of z is 1)
% Now the iid normal process e
[e_grid,pi_e]=discretizeAR1_FarmerToda(0,0,Params.sigma_epsilon_e,n_e);
e_grid=exp(e_grid); % Take exponential of the grid
pi_e=pi_e(1,:)'; % Because it is iid, the distribution is just the first row (all rows are identical). We use pi_e as a column vector for VFI Toolkit to handle iid variables.
mean_e=pi_e'*e_grid; % Because it is iid, pi_e is the stationary distribution (you could just use MarkovChainMoments(), I just wanted to demonstrate a handy trick)
e_grid=e_grid./mean_e; % Normalise the grid on z (so that the mean of e is 1)
% To use e variables we have to put them into the vfoptions and simoptions
vfoptions.n_e=n_e;
vfoptions.e_grid=e_grid;
vfoptions.pi_e=pi_e;
simoptions.n_e=vfoptions.n_e;
simoptions.e_grid=vfoptions.e_grid;
simoptions.pi_e=vfoptions.pi_e;

% Use divide-and-conquer and grid interpolation layer (see Life-Cycle Models 29 and 30)
% With two standard endogenous states, divide-and-conquer and grid interpolation layer are applied (only) to the first endogenous state.
vfoptions.divideandconquer=1; % turn on divide-and-conquer
vfoptions.gridinterplayer=1; % turn on grid interpolation layer
vfoptions.ngridinterp=20; % 20 evenly-spaced points between each pair of consecutive grid points on the first endogenous state
simoptions.gridinterplayer=vfoptions.gridinterplayer; % grid interpolation layer must also be set in simoptions (because it changes Policy size/interpretation)
simoptions.ngridinterp=vfoptions.ngridinterp;


% Grid for labour choice
h_grid=linspace(0,1,n_d)'; % Notice that it is imposing the 0<=h<=1 condition implicitly
% Switch into toolkit notation
d_grid=h_grid;

%% Now, create the return function
DiscountFactorParamNames={'beta','sj'};

% Use 'LifeCycleModel40_ReturnFn'
ReturnFn=@(h,a1prime,a2prime,a1,a2,z1,z2,w,sigma,psi,eta,agej,Jr,pension,r1,r2,adjcost,kappa_j,wg1,wg2,wg3,beta,sj)...
    LifeCycleModel40_ReturnFn(h,a1prime,a2prime,a1,a2,z1,z2,w,sigma,psi,eta,agej,Jr,pension,r1,r2,adjcost,kappa_j,wg1,wg2,wg3,beta,sj)

%% Solve the value function iteration problem
disp('Solve for Value fn and Policy fn using ValueFnIter command')
vfoptions.verbose=1;
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

% V is now (a,z,e,j). One dimension for each state variable.
% Compare
size(V)
% with
[n_a,n_z,n_e,N_j]
% there are the same.
% Policy is
size(Policy)
% which is the same as
[length(n_d)+length(n_a),n_a,n_z,n_e,N_j]
% The n_a,n_z,n_e,N_j represent the state on which the decisions/policys
% depend, and there is one decision for each decision variable 'd' and each
% endogenous state variable 'a', and one for the markov exogenous state variable
% 'z', and one for the markov exogenous state variable 'e'.

%% Testing divide-and-conquer
vfoptions.divideandconquer=0;
tic;
[V0, Policy0]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

vfoptions.divideandconquer=1;
vfoptions.level1n=[3,n_a(2)]; % DC2B
tic;
[V1, Policy1]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

vfoptions.divideandconquer=1;
vfoptions.level1n=[5,5]; % DC2
tic;
[V1, Policy1]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc


%% We won't plot the Value and Policy fn, but thinking out how you would might be a good way to check you understand the form of V and Policy

%% Now, we want to graph Life-Cycle Profiles

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_z,n_e],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,floor((n_z+1)/2),floor((n_e+1)/2))=1; % All agents start with zero assets, and the median value of each shock

%% We now compute the 'stationary distribution' of households
% Start with a mass of one at initial age, use the conditional survival
% probabilities sj to calculate the mass of those who survive to next
% period, repeat. Once done for all ages, normalize to one
Params.mewj=ones(1,Params.J); % Marginal distribution of households over age
for jj=2:length(Params.mewj)
    Params.mewj(jj)=Params.sj(jj-1)*Params.mewj(jj-1);
end
Params.mewj=Params.mewj./sum(Params.mewj); % Normalize to one
AgeWeightsParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (h,aprime,a,z) as first inputs, then just any relevant parameters.
FnsToEvaluate.fractiontimeworked=@(h,aprime,a,z,e) h; % h is fraction of time worked
FnsToEvaluate.earnings=@(h,aprime,a,z,e,w,kappa_j) w*kappa_j*h*z*e; % w*kappa_j*h*z*e is the labor earnings
FnsToEvaluate.assets=@(h,aprime,a,z,e) a; % a is the current asset holdings
% notice that we have called these fractiontimeworked, earnings and assets

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% For example
% AgeConditionalStats.earnings.Mean
% There are things other than Mean (Median, Gini, percentiles, etc.); in
% earlier deterministic models all agents were identical at each age so
% those were trivial, but now that we have an idiosyncratic shock z they
% are meaningful and worth looking at.

%% Plot the life cycle profiles of fraction-of-time-worked, earnings, and assets
figure(1)
subplot(3,1,1); plot(1:1:Params.J,AgeConditionalStats.fractiontimeworked.Mean)
title('Life Cycle Profile: Fraction Time Worked (h)')
subplot(3,1,2); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Labor Earnings (w kappa_j h)')
subplot(3,1,3); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Assets (a)')



