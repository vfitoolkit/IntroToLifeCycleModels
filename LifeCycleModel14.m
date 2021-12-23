%% Life-Cycle Model 14: More Life-Cycle Profiles
% We have previously looked at mean life-cycle profiles.
% But life-cycle profile are really just any model statistic conditional on age.
% So we look at some others, which were already automatically calculated, but in deterministic models were largely pointless to look at.
% We also look at how to group ages/periods together in bins, and report the life-cycle profiles conditional on these bins.
%
% The model is just that of Life-Cycle Model 9, and there is no change until line 154

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
Params.warmglow3=Params.sigma; % By using the same curvature as the utility of consumption it makes it much easier to guess appropraite parameter values for the warm glow

%% Grids
% The ^3 means that there are more points near 0 and near 10. We know from
% theory that the value function will be more 'curved' near zero assets,
% and putting more points near curvature (where the derivative changes the most) increases accuracy of results.
a_grid=10*(linspace(0,1,n_a).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.

% First, the AR(1) process z1
if Params.rho_z<0.99
    [z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_epsilon_z,n_z);
elseif Params.rho_z>=0.99 % Rouwenhourst performs better than Farmer-Toda when the autocorrelation is very high
    [z_grid,pi_z]=discretizeAR1_Rouwenhorst(0,Params.rho_z,Params.sigma_epsilon_z,n_z);
end
z_grid=exp(z_grid); % Take exponential of the grid
[mean_z,~,~,~]=MarkovChainMoments(z_grid,pi_z); % Calculate the mean of the grid so as can normalise it
z_grid=z_grid./mean_z; % Normalise the grid on z (so that the mean of z is 1)

% Grid for labour choice
h_grid=linspace(0,1,n_d)'; % Notice that it is imposing the 0<=h<=1 condition implicitly
% Switch into toolkit notation
d_grid=h_grid;

%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

% Notice we still use 'LifeCycleModel8_ReturnFn'
ReturnFn=@(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj) LifeCycleModel8_ReturnFn(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)

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
jequaloneDist=zeros(n_a,n_z,'gpuArray'); % Put no households anywhere on grid
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
FnsToEvaluate.fractiontimeworked=@(h,aprime,a,z) h; % h is fraction of time worked
FnsToEvaluate.earnings=@(h,aprime,a,z,w,kappa_j) w*kappa_j*z*h; % w*kappa_j*z*h is the labor earnings (note: h will be zero when z is zero, so could just use w*kappa_j*h)
FnsToEvaluate.assets=@(h,aprime,a,z) a; % a is the current asset holdings

% notice that we have called these fractiontimeworked, earnings and assets

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,[],Params,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% For example
% AgeConditionalStats.earnings.Mean
% There are things other than Mean, but in our current deterministic model
% in which all agents are born identical the rest are meaningless.

%% Plot the life cycle profiles of fraction-of-time-worked, earnings, and assets

figure(1)
subplot(3,1,1); plot(1:1:Params.J,AgeConditionalStats.fractiontimeworked.Mean)
title('Life Cycle Profile, Mean:Fraction Time Worked (h)')
subplot(3,1,2); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile, Mean: Labor Earnings (w kappa_j h)')
subplot(3,1,3); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile, Mean: Assets (a)')

%% Unchanged from Life-Cycle Model 9 until here (except deleting some graphs)
% Except minor change to titles of the Life Cycle Profiles ploted just a
% few lines above. So that it will be clearer the difference from what we plot next.

%% We have looked already at life-cycle profiles of the age-conditional mean.
% But lots of other things are being calculated.
% You can see them all be looking at, e.g.
AgeConditionalStats.earnings
% We see there are also: Median, Variance, LorenzCurve, Gini, QuantileCutoffs, and QuantileMeans

% So for example, we could plot the life-cycle profiles of the age-conditional variance
figure(2)
subplot(3,1,1); plot(1:1:Params.J,AgeConditionalStats.fractiontimeworked.Variance)
title('Life Cycle Profile, Variance: Fraction Time Worked (h)')
subplot(3,1,2); plot(1:1:Params.J,AgeConditionalStats.earnings.Variance)
title('Life Cycle Profile, Variance: Labor Earnings (w kappa_j h)')
subplot(3,1,3); plot(1:1:Params.J,AgeConditionalStats.assets.Variance)
title('Life Cycle Profile, Variance: Assets (a)')
xlabel('Age j')


% Notice that by default the 
% AgeConditionalStats.earnings.LorenzCurve
% has all 100 percentiles.

% We can very easily switch to just 10 deciles using the options
simoptions.npoints=10;
% We need to include options as the last input in
AgeConditionalStats_npoints10=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,[],Params,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);
% And we plot the decile means
figure(3)
subplot(3,1,1); plot(1:1:Params.J,AgeConditionalStats_npoints10.fractiontimeworked.QuantileMeans)
title('Life Cycle Profile, Decile Means: Fraction Time Worked (h)')
subplot(3,1,2); plot(1:1:Params.J,AgeConditionalStats_npoints10.earnings.QuantileMeans)
title('Life Cycle Profile, Decile Means: Labor Earnings (w kappa_j h)')
subplot(3,1,3); plot(1:1:Params.J,AgeConditionalStats_npoints10.assets.QuantileMeans)
title('Life Cycle Profile, Decile Means: Assets (a)')
xlabel('Age j')

% Let's switch back up to 100 points 
% (100 is default, I turn it back up as the number of points forms the basis of the calculation of the Gini coefficient.)
simoptions.npoints=100;
AgeConditionalStats2=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,[],Params,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% We can plot the life-cycle profiles of the age-conditional Gini coefficients
figure(4)
subplot(2,1,1); plot(1:1:Params.J,AgeConditionalStats2.earnings.Gini)
title('Life Cycle Profile, Gini: Labor Earnings (w kappa_j h)')
subplot(2,1,2); plot(1:1:Params.J,AgeConditionalStats2.assets.Gini)
title('Life Cycle Profile, Gini: Assets (a)')
xlabel('Age j')

% But these Gini coefficients for each single age don't tell us about
% inequality in the model economy as a whole. Let's instead look at the
% Gini for labor earnings among all working age people.

% We can set agegroupings to group ages into bins, lets first just do five year bins
simoptions.agegroupings=1:5:N_j;
AgeConditionalStats_5yrs=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,[],Params,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

figure(5)
subplot(2,1,1); plot(1:5:Params.J,AgeConditionalStats_5yrs.earnings.Gini)
title('Life Cycle Profile, Gini: Labor Earnings (w kappa_j h)')
subplot(2,1,2); plot(1:5:Params.J,AgeConditionalStats_5yrs.assets.Gini)
title('Life Cycle Profile, Gini: Assets (a)')
xlabel('Age j, grouped into 5 year age bins')

% We might also set a group of 'working age'
simoptions.agegroupings=[1,Params.Jr];
AgeConditionalStats_WorkingAgeRetired=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,[],Params,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% Note: The Gini coefficient of earnings of the retirees is calculated to be NaN
% because there is no variation in the earnings. I manually replace it with
% zero in the following graphs.
figure(6)
subplot(2,1,1); plot([1,Params.Jr],[AgeConditionalStats_WorkingAgeRetired.earnings.Gini(1),0])
title('Life Cycle Profile, Gini: Labor Earnings (w kappa_j h)')
subplot(2,1,2); plot([1,Params.Jr],AgeConditionalStats_WorkingAgeRetired.assets.Gini)
title('Life Cycle Profile, Gini: Assets (a)')





