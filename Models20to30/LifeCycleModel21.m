%% Life-Cycle Model 21: Idiosyncratic medical shocks in retirement
% In Life-Cycle Model 20 we saw how to make exogenous markov z have
% different transition probabilities at each age/period. We can also make
% the grid values differ by age (as long as number of grid points does not
% change). Because we only were using z as labor productivity during
% working ages, and then just ignoring z in retirement, we can just
% repurpose z during retirement to use it to model something else.
% That is what we do here, we use z as labor productivity during working
% ages, and as medical shocks in retirement. ExogShockFn is used to make
% both the grid and transition probailities depend on age.

% Note: Both the z_grid_J and the ExogShockFn approaches require that the
% number of grid points in z_grid does NOT change with age.

%% How does VFI Toolkit think about this?
%
% One decision variable: h, labour hours worked
% One endogenous state variable: a, assets (total household savings)
% One stochastic exogenous state variable: z, 'unemployment' shock
% Age: j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=51; % Endogenous labour choice (fraction of time worked)
n_a=201; % Endogenous asset holdings
n_z=2; % Exogenous labor productivity units shock
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

%% z_grid and pi_z as a function that depends on age
% Note that the dependence on age is just via the parameters that are input
% In this example, agej obviously depends on age
vfoptions.ExogShockFn=@(agej,Jr) LifeCycleModel21_ExogShockFn(agej,Jr);
simoptions.ExogShockFn=vfoptions.ExogShockFn;
% Both value function and simulations need to know about the age-dependence exogneous shocks

% We will evaluate the ExogShockFn at agej=1, just because I want to use
% the stationary distribtion as the initial distribution for agents.
[z_grid, pi_z]=vfoptions.ExogShockFn(1,Params.Jr);
% Note, because vfoptions.ExogShockFn exists the values of z_grid and pi_z
% are effectively ignored internally by the VFI Toolkit commands.

[mean_z,~,~,statdist_z]=MarkovChainMoments(z_grid,pi_z);

%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

% Now use 'LifeCycleModel21_ReturnFn'
ReturnFn=@(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj) LifeCycleModel21_ReturnFn(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
% vfoptions=struct(); % Just using the defaults.
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc
% Note: Because we have vfoptions.ExogShockFn, what we input for z_grid and  pi_z will just be ignored.

%% Now, we want to graph Life-Cycle Profiles

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are
% at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,:)=statdist_z; % All agents start with zero assets, and the median shock

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
% Note: Because we have simoptions.ExogShockFn, what we input for z_grid and  pi_z will just be ignored.

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (h,aprime,a,z) as first
% inputs, then just any relevant parameters.
FnsToEvaluate.fractiontimeworked=@(h,aprime,a,z) h; % h is fraction of time worked
FnsToEvaluate.earnings=@(h,aprime,a,z,w,kappa_j) w*kappa_j*z*h; % w*kappa_j*z*h is the labor earnings (note: h will be zero when z is zero, so could just use w*kappa_j*h)
FnsToEvaluate.assets=@(h,aprime,a,z) a; % a is the current asset holdings
FnsToEvaluate.fractionunemployed=@(h,aprime,a,z) (z==0); % indicator for z=0 (unemployment) [Note: only makes sense as unemployment for j=1,..,Jr
FnsToEvaluate.fractionwithmedicalexpenses=@(h,aprime,a,z) (z==0.3); % indicator for z=0.3 medical shock

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);
% Note: Because we have simoptions.ExogShockFn, what we input for z_grid will just be ignored.

% For example
% AgeConditionalStats.earnings.Mean
% There are things other than Mean, but in our current deterministic model
% in which all agents are born identical the rest are meaningless.

%% Plot the life cycle profiles of fraction-of-time-worked, earnings, and assets

figure(1)
subplot(5,1,1); plot(1:1:Params.J,AgeConditionalStats.fractiontimeworked.Mean)
title('Life Cycle Profile: Fraction Time Worked (h)')
subplot(5,1,2); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Labor Earnings (w kappa_j h)')
subplot(5,1,3); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Assets (a)')
subplot(5,1,4); plot(1:1:Params.J,[AgeConditionalStats.fractionunemployed.Mean(1:Params.Jr-1),nan(1,Params.J-Params.Jr+1)])
title('Life Cycle Profile: Fraction Unemployment (z==0)')
xlim([1,Params.J])
subplot(5,1,5); plot(1:1:Params.J,AgeConditionalStats.fractionwithmedicalexpenses.Mean)
title('Life Cycle Profile: Fraction experiencing medical expenses (z==0.5)')

% Notice how we only plot the first part of
% AgeConditionalStats.fractionunemployed.Mean(1:Params.Jr-1), becasuse this
% FnToEvaluate is based on z, which changes meaning between j=Jr-1 and j=Jr.

 
%% Solve the model again, but without medical shocks, to compare asset profiles
vfoptions=struct();
simoptions=struct();
% Turn of ExogShockFn by removing it from vfoptions and simoptions.
% Note that above we created z_grid and pi_z based on j=1
% So by using these now we will just be using those for all periods, which
% is exactly the model without medical shocks if we switch to the return
% function from Life-Cycle model 8.
ReturnFn=@(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj) LifeCycleModel8_ReturnFn(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)
[V_nomedical, Policy_nomedical]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
StationaryDist_nomedical=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy_nomedical,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
AgeConditionalStats_nomedical=LifeCycleProfiles_FHorz_Case1(StationaryDist_nomedical,Policy_nomedical,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

figure(2)
plot(1:1:Params.J,AgeConditionalStats.assets.Mean,1:1:Params.J,AgeConditionalStats_nomedical.assets.Mean)
title('Life Cycle Profile: Assets (a)')
legend('Medical Expense Shocks','No Medical Shocks')
% Notice that medical expense shocks late in life cause elderly households
% to hold more assets (as self-insurance against medical expense shocks)



