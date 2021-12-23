%% Life-Cycle Model 17: Precautionary Savings (with exogenous earnings)
% Stochastic shocks create a risk of hitting the borrowing constraint (as seen in Life-Cycle model 16). 
% Because households dislike hitting the borrowing contraint they take action to avoid it in the form 
% of higher savings, and this is called 'precautionary savings'. We will solve a model with exogenous 
% labor twice, once with exogenous shocks and once without. 

% We will see precautionary savings in two ways. First, we look directly at
% the policy functions for next period assets, and see how they differ when
% their are shocks. Second, we will look at life-cycle profiles and see how
% precautionary savings leads to higher average assets.

% We keep the expected value of earnings in the model with shocks to be exactly equal
% to that in the model without shocks, so the differences are driven by the existence of 
% the shocks, not that they change what happens on average.

% To better see the precautionary savings we largely disable other motives
% for savings by flattening the 

% Comment: Precautionary savings are much cleaner/more easily observed, in
% models with either just one period or with infinite periods. In a
% life-cycle model the uncertainty interacts with other aspects of the
% life-cycle in ways that serve to partially obscure it. It is however an
% important concept, especially when extending life-cycle models to general
% equilibrium.

%% How does VFI Toolkit think about this?
%
% No decision variable. Can set n_d=[], d_grid=[]
% One endogenous state variable: a, assets (total household savings)
% One stochastic exogenous state variable: z, an two-state process, a stochastic endowment representing employment and unemployment
% Age: j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=[]; % None
n_a=1001; % Endogenous asset holdings
n_z=3; % Exogenous labor productivity units shock
N_j=Params.J; % Number of periods in finite horizon

%% Parameters

% Discount rate
Params.beta = 0.96;
% Preferences
Params.sigma = 2; % Coeff of relative risk aversion (curvature of consumption)

% Prices
Params.w=1; % Wage
% Params.r=0.05; % Interest rate (0.05 is 5%)
Params.r=1/Params.beta-1; % If you change r to this then the consumption will be perfectly flat

% Demographics
Params.agej=1:1:Params.J; % Is a vector of all the agej: 1,2,3,...,J
Params.Jr=46;

% Pensions
Params.pension=0.8;

% Age-dependent labor productivity units
Params.kappa_j=[linspace(0.6,2,Params.Jr-15),linspace(2,1.5,14),zeros(1,Params.J-Params.Jr+1)];
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
Params.warmglow1=0.05; % (relative) importance of bequests
Params.warmglow2=0.3; % bliss point of bequests (essentially, the target amount)
Params.warmglow3=Params.sigma; % By using the same curvature as the utility of consumption it makes it much easier to guess appropraite parameter values for the warm glow

%% Grids
% The ^3 means that there are more points near 0 and near 10. We know from
% theory that the value function will be more 'curved' near zero assets,
% and putting more points near curvature (where the derivative changes the most) increases accuracy of results.
a_grid=10*(linspace(0,1,n_a).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.

% First, the AR(1) process z
[z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_epsilon_z,n_z);
z_grid=exp(z_grid); % Take exponential of the grid
[mean_z,~,~,statdist_z]=MarkovChainMoments(z_grid,pi_z); % Calculate the mean of the grid so as can normalise it
z_grid=z_grid./mean_z; % Normalise the grid on z (so that the mean of z is exactly 1)

d_grid=[]; % No decision variables

%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

% Use 'LifeCycleModel10_ReturnFn'
ReturnFn=@(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj) LifeCycleModel10_ReturnFn(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
vfoptions=struct(); % Just using the defaults.
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

%% Now, we want to graph Life-Cycle Profiles

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are
% at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,:)=statdist_z; % All agents start with zero assets, and the stationary distribution of shocks

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
FnsToEvaluate.earnings=@(aprime,a,z,w,kappa_j) w*kappa_j*z; % z is the 'stochastic endowment' or 'exogenous earnings'
FnsToEvaluate.assets=@(aprime,a,z) a; % a is the current asset holdings
FnsToEvaluate.consumption=@(aprime,a,z,agej,Jr,w,kappa_j,r,pension) (agej<Jr)*(w*kappa_j*z+(1+r)*a-aprime)+(agej>=Jr)*(pension+(1+r)*a-aprime);

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,[],Params,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

%% Now let's just solve the model again, but this time without the exogenous shocks
% Note that we normalized the mean of the shocks to one, so replacing them
% with a value of one with probability one has no impact on mean earnings.

n_z_noshock=1;
z_grid_noshock=1;
pi_z_noshock=1;
% Run the parts of the model that change
[V_noshock, Policy_noshock]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z_noshock,N_j, d_grid, a_grid, z_grid_noshock, pi_z_noshock, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
jequaloneDist=zeros(n_a,n_z_noshock,'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,1)=1; % All agents start with zero assets, and the median shock
StationaryDist_noshock=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy_noshock,n_d,n_a,n_z_noshock,N_j,pi_z_noshock,Params,simoptions);
% Calculate the same life-cycle profiles, but without shock
AgeConditionalStats_noshock=LifeCycleProfiles_FHorz_Case1(StationaryDist_noshock,Policy_noshock,FnsToEvaluate,[],Params,n_d,n_a,n_z_noshock,N_j,d_grid,a_grid,z_grid_noshock,simoptions);

%% Compare the next period asset policy with and without shocks to see 'precautionary savings'

PolicyVals=PolicyInd2Val_FHorz_Case1(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid);
PolicyVals_noshock=PolicyInd2Val_FHorz_Case1(Policy_noshock,n_d,n_a,n_z_noshock,N_j,d_grid,a_grid);

% Plot the aprime policy as a function (of assets) for a given age  (I do a few for different ages)
zind=floor(n_z+1)/2; % This will be the median
figure(1)
subplot(2,1,1); plot(a_grid,PolicyVals(1,:,1,1),a_grid,PolicyVals(1,:,zind,1),a_grid,PolicyVals(1,:,end,1),a_grid,PolicyVals_noshock(1,:,1,1)) % j=1
title('Policy for aprime at age j=1 (at low asset levels)')
legend('min z','median z','max z','no shock')
xlim([0,0.1])
subplot(2,1,2); plot(a_grid,PolicyVals(1,:,1,20),a_grid,PolicyVals(1,:,zind,20),a_grid,PolicyVals(1,:,end,20),a_grid,PolicyVals_noshock(1,:,1,20)) % j=20
title('Policy for aprime at age j=20 (at low asset levels)')
xlim([0,0.1])
% Precautionary savings can been seen by how households with the median
% shock save more than those with no shock (these are the precautionary
% savings at low asset levels).
% This is despite the fact that the median grid point with shock is 0.9976,
% so actually just below the no shock case where z=1. That is, households
% are saving more, due to precautionary savings, despite having lower
% earnings.
% Note: have zoomed in on policies of households low assets holdings (on x-axis near zero)

figure(2)
subplot(3,1,1); plot(a_grid,PolicyVals(1,:,1,45),a_grid,PolicyVals(1,:,zind,45),a_grid,PolicyVals(1,:,end,45),a_grid,PolicyVals_noshock(1,:,1,45)) % j=45
title('Policy for aprime at age j=45 (at low asset levels)')
xlim([0,0.1])
legend('min z','median z','max z','no shock')
subplot(3,1,2); plot(a_grid,PolicyVals(1,:,1,46),a_grid,PolicyVals(1,:,zind,46),a_grid,PolicyVals(1,:,end,46),a_grid,PolicyVals_noshock(1,:,1,46)) % j=46
title('Policy for aprime at age j=46 (first year of retirement)')
subplot(3,1,3); plot(a_grid,PolicyVals(1,:,1,81),a_grid,PolicyVals(1,:,zind,81),a_grid,PolicyVals(1,:,end,81),a_grid,PolicyVals_noshock(1,:,1,81)) % j=81
title('Policy for aprime at age j=81')
xlabel('Assets (a)')
% Note: First panel shows the period before retirement, so there will be no
% shocks in the future. Now households with the median shock save less than
% those for no shock because there is no precautionary motive and their
% earnings are fractionally lower.
% Note: Once households reach retirement there are no longer any relevant shocks, hence they behave identically


%% Compare the life-cycle profiles of assets to see precautionary savings

figure(3)
subplot(4,2,1); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile, AR(1) shock: Labor Earnings (kappa_j*z)')
subplot(4,2,3); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile, AR(1) shock: Assets (a)')
subplot(4,2,5); plot(1:1:Params.J,AgeConditionalStats.consumption.Mean)
title('Life Cycle Profile, AR(1) shock: Consumption (c)')
subplot(4,2,2); plot(1:1:Params.J,AgeConditionalStats_noshock.earnings.Mean)
title('Life Cycle Profile, no shock: Labor Earnings (kappa_j*z)')
subplot(4,2,4); plot(1:1:Params.J,AgeConditionalStats_noshock.assets.Mean)
title('Life Cycle Profile, no shock: Assets (a)')
subplot(4,2,6); plot(1:1:Params.J,AgeConditionalStats_noshock.consumption.Mean)
title('Life Cycle Profile, no shock: Consumption (c)')
subplot(4,2,7); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean-AgeConditionalStats_noshock.earnings.Mean)
title('Life Cycle Profile, shock minus no shock: Labor Earnings (kappa_j*z)')
ylim([-0.01,0.01])
subplot(4,2,8); plot(1:1:Params.J,AgeConditionalStats.assets.Mean-AgeConditionalStats_noshock.assets.Mean)
title('Life Cycle Profile, shock minus no shock: Assets (a)')
% Notice how there is no difference in labor earnings (bottom left panel).
% By contrast there are more assets in the model with shocks.
% Note also the interaction with the life-cycle, early in life agents all
% have zero assets (would have negative if they could) due to life-cycle
% consumption smoothing vs hump-shaped earnings motives we discussed in
% Life-Cycle model 15. These overwhelm the precautionary savings motive for
% these early ages. Only later when agents would typically hold assets (see
% the life-cycle profiles of mean assets in the second row) are the
% precautionary savings motives swamped by consumption-smoothing motives
% and lead to precautionary savings seen in bottom right panel.


%% Aggregate Variables
% This is not something we have looked at before because it does not make a
% lot of sense in a life-cycle model. It is more for OLG models (combining
% this household with a 'whole economy' in the form of firms and general equilibrium).
AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z,N_j, d_grid, a_grid, z_grid,[],simoptions);
% AggVars are 'aggregate variables' and sum up the FnsToEvaluate across the
% whole stationary distribution (in the theory they are integrals not sums).

AggVars_noshock=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist_noshock, Policy_noshock, FnsToEvaluate, Params, [], n_d, n_a, n_z_noshock,N_j, d_grid, a_grid, z_grid_noshock,[],simoptions);

% Total assets is the asset holdings of all households summed up across the stationary distribution.
fprintf('Total assets by all households, with shocks, are equal to %8.4f \n',AggVars.assets.Mean)
fprintf('Total assets by all households, no shocks, are equal to %8.4f \n',AggVars_noshock.assets.Mean)
fprintf('Precautionary savings leads to more assets in model with shocks \n')








