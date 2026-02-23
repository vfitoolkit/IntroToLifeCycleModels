%% Life-Cycle Model 37: Temptation and Self-Control, Gul-Pesendorfer preferences
% Alter life-cycle model 10 to use Gul-Pesendorfer preferences

% There are essentially just one aspect to using Gul-Pesendorfer
% 1. Set up the temptation function, vfoptions.temptationFn, (and associated parameters)

% 1. The temptation parameters are
Params.sigmatempt=2; % curvature of consumption (note, is set to same value as sigma)
Params.scaletemptation=1; % make is so temptation is same as standard utilty (this will substantially reduce assets in a visibly obvious way)

vfoptions.exoticpreferences='GulPesendorfer'; % Use Gul-Pesendorfer preferences

% That is all. Every other line of code is unchanged!! Temptation really is that easy ;)

%% How does VFI Toolkit think about this?
%
% One endogenous state variable: a, assets (total household savings)
% One stochastic exogenous state variable: z, an AR(1) process (in logs), idiosyncratic shock to labor earnings
% Age: j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=0;
n_a=201; % Endogenous asset holdings
n_z=21; % Exogenous labor productivity units shock
N_j=Params.J; % Number of periods in finite horizon

%% Parameters

% Discount rate
Params.beta = 0.96;
% Preferences
Params.sigma = 2; % Coeff of relative risk aversion (curvature of consumption)

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


%% Grids
% The ^3 means that there are more points near 0 and near 10. We know from
% theory that the value function will be more 'curved' near zero assets,
% and putting more points near curvature (where the derivative changes the most) increases accuracy of results.
a_grid=10*(linspace(0,1,n_a(1)).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.

% First, the AR(1) process z
[z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_epsilon_z,n_z);
z_grid=exp(z_grid); % Take exponential of the grid
[mean_z,~,~,~]=MarkovChainMoments(z_grid,pi_z); % Calculate the mean of the grid so as can normalise it
z_grid=z_grid./mean_z; % Normalise the grid on z (so that the mean of z is exactly 1)

d_grid=[];


%% Set up the temptation function in terms of (d,aprime,a,z) (this model does not have d), doing this is a lot like just a copy of the return function
vfoptions.temptationFn=@(aprime,a,z,w,sigmatempt,scaletemptation,agej,Jr,pension,r,kappa_j) LifeCycleModel37_TemptationFn(aprime,a,z,w,sigmatempt,scaletemptation,agej,Jr,pension,r,kappa_j);

%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

% LifeCycleModel37_ReturnFn is essentially just LifeCycleModel10_ReturnFn without the warm-glow
ReturnFn=@(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j) ...
    LifeCycleModel37_ReturnFn(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j);

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc


%% Let's take a quick look at what we have calculated, namely V and Policy

% The value function V depends on the state, so now it depends on both asset holdings and age.

% We can plot V as a 3d plot (surf is matlab command for 3d plot)
% Which z value should we plot? I will plot the median
zind=floor(n_z+1)/2; % This will be the median
figure(1)
subplot(2,1,1); surf(a_grid*ones(1,Params.J),ones(n_a,1)*(1:1:Params.J),reshape(V(:,zind,:),[n_a,Params.J]))
title('Value function: median value of z')
xlabel('Age j')
ylabel('Assets (a)')
subplot(2,1,2); surf(a_grid*ones(1,Params.J),ones(n_a,1)*(Params.agejshifter+(1:1:Params.J)),reshape(V(:,zind,:),[n_a,Params.J]))
title('Value function: median value of z')
xlabel('Age in Years')
ylabel('Assets (a)')

% Do another plot of V, this time as a function (of assets) for a given age (I do a few for different ages)
figure(2)
subplot(5,1,1); plot(a_grid,V(:,1,1),a_grid,V(:,zind,1),a_grid,V(:,end,1)) % j=1
title('Value fn at age j=1')
legend('min z','median z','max z') % Just include the legend once in the top subplot
subplot(5,1,2); plot(a_grid,V(:,1,20),a_grid,V(:,zind,20),a_grid,V(:,end,20)) % j=20
title('Value fn at age j=20')
subplot(5,1,3); plot(a_grid,V(:,1,45),a_grid,V(:,zind,45),a_grid,V(:,end,45)) % j=45
title('Value fn at age j=45')
subplot(5,1,4); plot(a_grid,V(:,1,46),a_grid,V(:,end,46),a_grid,V(:,end,46)) % j=46
title('Value fn at age j=46 (first year of retirement)')
subplot(5,1,5); plot(a_grid,V(:,1,81),a_grid,V(:,zind,81),a_grid,V(:,end,81)) % j=81
title('Value fn at age j=81')
xlabel('Assets (a)')

% Convert the policy function to values (rather than indexes).
% Note that there is one policy for next period assets (aprime). 
% Policy(1,:,:,:) is aprime [as function of (a,z,j)]
% Plot as a 3d plot, again I arbitrarily choose the median value of z
figure(3)
PolicyVals=PolicyInd2Val_Case1_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions);
surf(a_grid*ones(1,Params.J),ones(n_a,1)*(1:1:Params.J),reshape(PolicyVals(1,:,zind,:),[n_a,Params.J]))
title('Policy function: next period assets (aprime), median z')
xlabel('Age j')
ylabel('Assets (a)')
zlabel('Next period assets (aprime)')

% Again, plot policy (aprime), this time as a function (of assets) for a given age  (I do a few for different ages)
figure(4)
subplot(5,1,1); plot(a_grid,PolicyVals(1,:,1,1),a_grid,PolicyVals(1,:,zind,1),a_grid,PolicyVals(1,:,end,1)) % j=1
title('Policy for aprime at age j=1')
legend('min z','median z','max z') % Just include the legend once in the top-right subplot
subplot(5,1,2); plot(a_grid,PolicyVals(1,:,1,20),a_grid,PolicyVals(1,:,zind,20),a_grid,PolicyVals(1,:,end,20)) % j=20
title('Policy for aprime at age j=20')
subplot(5,1,3); plot(a_grid,PolicyVals(1,:,1,45),a_grid,PolicyVals(1,:,zind,45),a_grid,PolicyVals(1,:,end,45)) % j=45
title('Policy for aprime at age j=45')
subplot(5,1,4); plot(a_grid,PolicyVals(1,:,1,46),a_grid,PolicyVals(1,:,zind,46),a_grid,PolicyVals(1,:,end,46)) % j=46
title('Policy for aprime at age j=46 (first year of retirement)')
subplot(5,1,5); plot(a_grid,PolicyVals(1,:,1,81),a_grid,PolicyVals(1,:,zind,81),a_grid,PolicyVals(1,:,end,81)) % j=81
title('Policy for aprime at age j=81')
xlabel('Assets (a)')

%% Now, we want to graph Life-Cycle Profiles

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are
% at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,floor((n_z+1)/2))=1; % All agents start with zero assets, and the median shock

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
% Again, we will explain in a later model what the stationary distribution
% is, it is not important for our current goal of graphing the life-cycle profile

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (h,aprime,a,z) as first
% inputs, then just any relevant parameters.
FnsToEvaluate.earnings=@(aprime,a,z,w,kappa_j) w*kappa_j*z; % w*kappa_j*z*h is the labor earnings (note: h will be zero when z is zero, so could just use w*kappa_j*h)
FnsToEvaluate.assets=@(aprime,a,z) a; % a is the current asset holdings

% notice that we have called these fractiontimeworked, earnings and assets

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% For example
% AgeConditionalStats.earnings.Mean
% There are things other than Mean, but in our current deterministic model
% in which all agents are born identical the rest are meaningless.

%% Plot the life cycle profiles of fraction-of-time-worked, earnings, and assets

figure(5)
subplot(2,1,1); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Labor Earnings (w kappa_j h)')
subplot(2,1,2); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Assets (a)')

