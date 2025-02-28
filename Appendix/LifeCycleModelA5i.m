%% Life-Cycle Model A5i: Two markov (z) shocks, independent
% Use two exogenous shocks: z1 and z2, both are shocks to labor efficiency units
% We have to change ReturnFn and FnsToEvaluate inputs to take (...,z1,z2,...)
% We start with a simple example where we just have two independent markovs.
% Both z1 and z2 are AR(1), and as in Life-Cycle Model 9, we just use
% Tanaka-Toda method to discretize them which gives us z1_grid, z2_grid,
% pi_z1 and pi_z2.
% 
% We need to put together z1_grid and z2_grid to get z_grid
% Note that z1_grid and z2_grid are both column-vectors
% We put them together by 'stacking' the column-vectors on top of each other.
% z_grid=[z1_grid; z2_grid];
% So z_grid is a 'stacked column vector' representation of the values for z1 and z2.
%
% We also need to put together pi_z1 and pi_z2 to get pi_z.
% We do this as
% pi_z=kron(pi_z2,pi_z1);
% Notice that this is the kron(), with pi_z1 and pi_z2 in reverse order.
%
% So we have size(pi_z)=sum(n_z)-by-1 and size(pi_z)=prod(n_z)-by-prod(n_z)
%
% How to interpret pi_z?
% Say z1 has three values [z1a;z1b;z1c], and z2 has two values [z2a; z2b]
% Consider putting them together, so that we count through all the z1
% values, keeping z2 value fixed, and once we complete all z1 values we
% incrememnt z2 value and start counting through z1 values again.
% So think of
% [z1a, z2a;
%  z1b, z2a;
%  z1c, z2a;
%  z1a, z2b;
%  z1b, z2b;
%  z1c, z2b]
% This is what the dimensions of pi_z are capturing. The rows are this
% period and the columns are next period (as always for our markov
% transition matrices).
% So for example the 2nd row, 3rd column element in pi_z represents the
% probability of going from (z1b,z2a) today, to (z1c, z2a) tomorrow.
% Another example, the 5th row, 2nd column element in pi_z represents the
% probability of going from (z1b,z2b) today, to (z1b, z2a) tomorrow.
%
% These three things:
% i) change inputs of ReturnFn and FnsToEvaluate to include (...,z1,z2,...)
% ii) create z_grid (here as a stacked column as they are independent)
% iii) create pi_z (here as kron() of reverse order, as they are independent)
% are the only changes we need to make to the code.
%
% The following Life-Cycle Models A5ii, A5iii, A5iv consider more advanced
% approaches where the grids and transition probabilites for z1 and z2
% might be related (say because they are correlated). In all of these it is
% just about changing how we do steps (ii) and (iii).
%
% Note: The above explains z_grid and pi_z with 3 points on z1 and 2 points
% on z2. The code below uses 5 points on z1 and 3 points on z2. Take a look
% at z_grid and pi_z, see if they make sense to you (what size they are,
% and how to interpret them).

%% How does VFI Toolkit think about this?
%
% One decision variable: h, labour hours worked
% One endogenous state variable: a, assets (total household savings)
% Two stochastic exogenous state variables: z1 and z2, both are shocks to labor efficiency units
% Age: j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=51; % Endogenous labour choice (fraction of time worked)
n_a=201; % Endogenous asset holdings
n_z=[5,3]; % Exogenous labor productivity units shock, two of them
N_j=Params.J; % Number of periods in finite horizon

%% Parameters

% Discount rate
Params.beta = 0.96;
% Preferences
Params.sigma = 2; % Coeff of relative risk aversion (curvature of consumption)
Params.eta = 1.5; % Curvature of leisure (This will end up being 1/Frisch elasty)
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
% Age-dependent labor productivity units
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
% Exogenous shock process, z1: AR1 on labor productivity units
Params.rho_z1=0.9;
Params.sigma_epsilon_z1=0.03;
% Exogenous shock process, z2: AR1 on labor productivity units
Params.rho_z2=0.3;
Params.sigma_epsilon_z2=0.01;

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
Params.warmglow3=Params.sigma; % By using the same curvature as the utility of consumption it makes it much easier to guess appropraite parameter values for the warm glow

%% Grids
% The ^3 means that there are more points near 0 and near 10. We know from
% theory that the value function will be more 'curved' near zero assets,
% and putting more points near curvature (where the derivative changes the most) increases accuracy of results.
a_grid=10*(linspace(0,1,n_a).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.

% Grid for labour choice
h_grid=linspace(0,1,n_d)'; % Notice that it is imposing the 0<=h<=1 condition implicitly
% Switch into toolkit notation
d_grid=h_grid;

%% Dealing with two markovs: grid and transition matrix, z_grid and pi_z
% First, just discretize each of our two AR(1) processes
% Discretize the AR(1) process z1
[z1_grid,pi_z1]=discretizeAR1_FarmerToda(0,Params.rho_z1,Params.sigma_epsilon_z1,n_z(1));
z1_grid=exp(z1_grid); % Take exponential of the grid
[mean_z1,~,~,~]=MarkovChainMoments(z1_grid,pi_z1); % Calculate the mean of the grid so as can normalise it
z1_grid=z1_grid./mean_z1; % Normalise the grid on z1 (so that the mean of z1 is exactly 1)

% Discretize the AR(1) process z2
[z2_grid,pi_z2]=discretizeAR1_FarmerToda(0,Params.rho_z2,Params.sigma_epsilon_z2,n_z(2));
z2_grid=exp(z2_grid); % Take exponential of the grid
[mean_z2,~,~,~]=MarkovChainMoments(z2_grid,pi_z2); % Calculate the mean of the grid so as can normalise it
z2_grid=z2_grid./mean_z2; % Normalise the grid on z2 (so that the mean of z2 is exactly 1)

% Now, we put together the two grids, as a stacked column
z_grid=[z1_grid; z2_grid];

% Next, we put together pi_z
pi_z=kron(pi_z2,pi_z1); % note reverse order


%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

% Use 'LifeCycleModelA5_ReturnFn', which has two markov exogenous states
ReturnFn=@(h,aprime,a,z1,z2,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)... 
    LifeCycleModelA5_ReturnFn(h,aprime,a,z1,z2,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
vfoptions=struct(); % Just using the defaults.
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

% V is now (a,z1,z2,j). One dimension for each state variable.
% Compare
size(V)
% with
[n_a,n_z(1),n_z(2),N_j]
% there are the same.
% Policy is
size(Policy)
% which is the same as
[length(n_d)+length(n_a),n_a,n_z(1),n_z(2),N_j]
% The n_a,n_z(1),n_z(2),N_j represent the state on which the decisions/policys
% depend, and there is one decision for each decision variable 'd' and each
% endogenous state variable 'a', and one for each exogenous state variable 'z'

%% We won't plot the Value and Policy fn, but thinking out how you would might be a good way to check you understand the form of V and Policy

%% Now, we want to graph Life-Cycle Profiles

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are
% at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,floor((n_z(1)+1)/2),floor((n_z(2)+1)/2))=1; % All agents start with zero assets, and the median shock

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
simoptions=struct(); % Use the default options
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);


%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (h,aprime,a,z) as first
% inputs, then just any relevant parameters.
FnsToEvaluate.fractiontimeworked=@(h,aprime,a,z1,z2) h; % h is fraction of time worked
FnsToEvaluate.earnings=@(h,aprime,a,z1,z2,w,kappa_j) w*kappa_j*h*z1*z2; % w*kappa_j*h*z is the labor earnings (note: h will be zero when z is zero, so could just use w*kappa_j*h)
FnsToEvaluate.assets=@(h,aprime,a,z1,z2) a; % a is the current asset holdings

% notice that we have called these fractiontimeworked, earnings and assets

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% For example
% AgeConditionalStats.earnings.Mean
% There are things other than Mean, but in our current deterministic model
% in which all agents are born identical the rest are meaningless.

%% Plot the life cycle profiles of fraction-of-time-worked, earnings, and assets

figure(1)
subplot(3,1,1); plot(1:1:Params.J,AgeConditionalStats.fractiontimeworked.Mean)
title('Life Cycle Profile: Fraction Time Worked (h)')
subplot(3,1,2); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Labor Earnings (w kappa_j h)')
subplot(3,1,3); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Assets (a)')



