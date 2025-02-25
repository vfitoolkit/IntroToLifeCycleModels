%% Life-Cycle Model 22: Deterministic income growth 
% (with endogneous labor supply and exogenous shocks)
% We will consider a (real wage) growth rate of 2%, Params.g=0.02
%
% We first solve the 'renormalized model'; see the pdf for an explanation.
% There are essentially three changes involved: the discount factor and the
% budget constraint, and the utility function.
% [Changing the utility function is needed to change from seperable to non-seperable utility]
% We then simulate panel data using the renormalized model, just as we normally would.
% We then adjust the panel data adding back in the growth that we had renormalized away.
%
% When defining functions to evaluate and simulating panel data I am going
% to refer to the renormalized variables as, e.g., ahat for a, what for w,
% etc. (Just like in the pdf)

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
Params.sigma1 = 0.3; % Weight on consumption
Params.sigma2 = 2;  % Curvature of utility
% Note: I have not thought seriously about appropriate parameter values for sigma1, sigma2. These are very arbitrary

% Deterministic wage growth rate
Params.g=0.02;

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
Params.warmglow1=0.3; % (relative) importance of bequests
Params.warmglow2=3; % bliss point of bequests (essentially, the target amount)
Params.warmglow3=2; % Curvature of warm-glow (conrols rate at which marginal utility diminishes)


%% Grids
% The ^3 means that there are more points near 0 and near 10. We know from
% theory that the value function will be more 'curved' near zero assets,
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
% Define the additional discount factor
Params.growthdiscount=(1+Params.g)^(Params.sigma1*(1-Params.sigma2));
DiscountFactorParamNames={'beta','sj','growthdiscount'};

% Change to 'LifeCycleModel22_ReturnFn'
ReturnFn=@(h,aprime,a,z,w,sigma1,sigma2,g,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj) LifeCycleModel22_ReturnFn(h,aprime,a,z,w,sigma1,sigma2,g,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
vfoptions=struct(); % Just using the defaults.
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

%% Now, we want simulate panel data

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are
% at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,floor((n_z+1)/2))=1; % All agents start with zero assets, and the median shock

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (h,aprime,a,z) as first
% inputs, then just any relevant parameters.
FnsToEvaluate.fractiontimeworked=@(h,aprime,a,z) h; % h is fraction of time worked
FnsToEvaluate.earningshat=@(h,aprime,a,z,w,kappa_j) w*kappa_j*z*h; % w*kappa_j*z*h is the labor earnings (note: h will be zero when z is zero, so could just use w*kappa_j*h)
FnsToEvaluate.ahat=@(h,aprime,a,z) a; % a is the current asset holdings

%% Simulate panel data
% Note: It is not presently a feature of SimPanelValues to account for
% death ('conditional survival probabilities). Simulations will be of
% different lengths as individuals are drawn from the stationary
% distribution and so most will reach the final period after which all
% entries are nan. (If this feature would be useful to you, please email
% me: robertdkirkby@gmail.com and I can implement it)

% To simulate panel data we will set the number of time periods
simoptions.simperiods=N_j; % N_j is the default value
simoptions.numbersims=10^3; % 10^3 is the default value
% To simulate panel data you have to define 'where' an individual household
% simulation starts from, we will use the StationaryDist (from which
% starting points will be drawn randomly)
InitialDist=jequaloneDist;

SimPanelValues=SimPanelValues_FHorz_Case1(InitialDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,pi_z, simoptions);
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length 'simperiods'

% For example
% SimPanelValues.earnings
% is the simulate panel of FnsToEvaluate.earnings
% size(SimPanelValues.earnings) is [simperiods,numbersims] 
% (what econometric theory on panel data would typically call T-by-N)

% Lets draw the time series plots of h, earnings and assets for ten households (arbirarily, the first ten)
figure(1)
subplot(3,1,1); plot(1:1:Params.J,SimPanelValues.fractiontimeworked(:,16)) % Note that we set simperiod so to be of lenght J (which would anyway have been the default)
title('Renormalized Time Series of one Household: Fraction Time Worked (h)')
subplot(3,1,2); plot(1:1:Params.J,SimPanelValues.earningshat(:,16))
title('Renormalized Time Series of one Household: Labor Earnings (what kappa_j h)')
subplot(3,1,3); plot(1:1:Params.J,SimPanelValues.ahat(:,16))
title('Renormalized Time Series of one Household: Assets (ahat)')

%% We can get simulated panel data for the original model using our the equations relating
% a with ahat, and w with what. Let's do this and plot the same time series
% as in figure 1, but this time for the original model rather than the renormalized model.

% First, notice that there is no change to FnsToEvaluate.fractiontimeworked

% For earningshat, we need to adjust it, multiplying what by (1+g)^j to get w (and hence changing from earningshat to earnings)
SimPanelValues.earnings=SimPanelValues.earningshat .* ((1+Params.g).^(1:1:simoptions.simperiods))';
% Note: this formula assumes that InitialDist is all age j=1 (it is in the current script, but need not be in general)

% For ahat, we need to adjust it, multiplying ahat by (1+g)^j to get a
SimPanelValues.a=SimPanelValues.ahat .* ((1+Params.g).^(1:1:simoptions.simperiods))';
% Note: this formula assumes that InitialDist is all age j=1 (it is in the current script, but need not be in general)

% Redraw the exact same plot, but now in the original model with wage growth
% Lets draw the time series plots of h, earnings and assets for ten households (arbirarily, the first ten)
figure(2)
subplot(3,1,1); plot(1:1:Params.J,SimPanelValues.fractiontimeworked(:,16)) % Note that we set simperiod so to be of lenght J (which would anyway have been the default)
title('Time Series of one Household: Fraction Time Worked (h)')
subplot(3,1,2); plot(1:1:Params.J,SimPanelValues.earnings(:,16))
title('Time Series of one Household: Labor Earnings (w kappa_j h)')
subplot(3,1,3); plot(1:1:Params.J,SimPanelValues.a(:,16))
title('Time Series of one Household: Assets (a)')


