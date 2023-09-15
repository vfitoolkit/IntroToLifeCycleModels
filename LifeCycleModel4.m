%% Life-Cycle Model 4: Life-Cycle Profiles
% We make no changes to the model.
% We just want to look at how we can create 'life-cycle profiles'
% Life-cycle profiles are a variable conditional on age, in our case it will be the conditional mean of three 'functions to evaluate'.
% This code is unchanged from Life-Cycle Model 3 until line 74.

%% How does VFI Toolkit think about this?
%
% One decision variable: h, labour hours worked
% One endogenous state variable: a, assets (total household savings)
% No stochastic exogenous state variables
% Age: j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=51; % Endogenous labour choice (fraction of time worked)
n_a=201; % Endogenous asset holdings
n_z=0; % This is how the VFI Toolkit thinks about deterministic models
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

%% Grids
% The ^3 means that there are more points near 0 and near 10. We know from
% theory that the value function will be more 'curved' near zero assets,
% and putting more points near curvature (where the derivative changes the most) increases accuracy of results.
a_grid=10*(linspace(0,1,n_a).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.

% Grid for labour choice
h_grid=linspace(0,1,n_d)'; % Notice that it is imposing the 0<=h<=1 condition implicitly
% Switch into toolkit notation
d_grid=h_grid;

%% Now, create the return function 
DiscountFactorParamNames={'beta'};

% Notice still using 'LifeCycleModel3_ReturnFn'
ReturnFn=@(h,aprime,a,w,sigma,psi,eta,agej,Jr,pension,r) LifeCycleModel3_ReturnFn(h,aprime,a,w,sigma,psi,eta,agej,Jr,pension,r)

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
vfoptions=struct(); % Just using the defaults.
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, [], [], ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

%% Until now the code is unchanged.

%% Now, we want to graph Life-Cycle Profiles

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are
% at age j=1. We will give them all zero assets.
jequaloneDist=zeros(n_a,1,'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1)=1; % Note that 0 is the 1st grid point in the asset grid
% We have put all the 'new' households (mass of 1) here (zero assets)

%% We now compute the 'stationary distribution' of households
% This is effectively irrelevant to understanding life-cycle profiles but it is required as an input. 
% We will explain in a later model what the stationary distribution of households is and what we are doing here.
% We need to say how many agents are of each age (this is needed for the
% stationary distribution but is actually irrelevant to the life-cycle profiles)
Params.mewj=ones(1,Params.J)/Params.J; % Put a fraction 1/J at each age
AgeWeightsParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age
simoptions=struct(); % Use the default options
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,[],Params,simoptions);
% Again, we will explain in a later model what the stationary distribution
% is, it is not important for our current goal of graphing the life-cycle profile

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (h,aprime,a) as first
% inputs, then just any relevant parameters.
FnsToEvaluate.fractiontimeworked=@(h,aprime,a) h; % h is fraction of time worked
FnsToEvaluate.earnings=@(h,aprime,a,w) w*h; % w*h is the labor earnings
FnsToEvaluate.assets=@(h,aprime,a) a; % a is the current asset holdings

% notice that we have called these fractiontimeworked, earnings and assets

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,[],Params,n_d,n_a,n_z,N_j,d_grid,a_grid,[],simoptions);

% For example
% AgeConditionalStats.earnings.Mean
% There are things other than Mean, but in our current deterministic model
% in which all agents are born identical the rest are meaningless.

%% Plot the life cycle profiles of fraction-of-time-worked, earnings, and assets

figure(1)
subplot(3,1,1); plot(1:1:Params.J,AgeConditionalStats.fractiontimeworked.Mean)
title('Life Cycle Profile: Fraction Time Worked (h)')
subplot(3,1,2); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Labor Earnings (wh)')
subplot(3,1,3); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Assets (a)')



