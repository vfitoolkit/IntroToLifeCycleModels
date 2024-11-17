%% Life-Cycle Model 45: GMM Estimation of a Life-Cycle Model with Permanent Types, using the Permanent Types for Unobserved Heterogeneity
% We will make one change to the model used in Life-Cycle Model 45 (which was essentially just Life-Cycle Model 9)
% We use permanent types as a fixed-effect in earnings (conditional on working)
% To be able to estimate them, we parameterize these fixed effects as a normal distribution.

% To keep things clear, we will not use any real-world data (that is done in Life-Cycle Model 47)
% We will first solve the model to get age-conditional mean and variance of earnings.
% Then we estimate the parameters of the normal distribution of the permanent type, 
% with these age-conditional mean and variance of earnings as the targets.

% In a model with permament types normally we would use the parameter named
% in PTypeDistParamNames, with value in Params, to set the distribution of permanent types.
% [The other option is to include permanent types as a dimension of the initial distribution]
% We need to set PTypeDistParamNames as a function, so that the distribution can be estimated. To keep
% things nice we will parametrize the permanent types as a normal distribution.

% Note: in practice you would not be able to tell the mean of the normal
% distribution of the fixed-effect apart from kappa_j, and so you would
% just set logalpha_mean=0. But here we have fixed kappa_j, so we can estimate
% logalpha_mean even though in practice this would be an odd thing to do.

%% How does VFI Toolkit think about this?
%
% One decision variable: h, labour hours worked
% One endogenous state variable: a, assets (total household savings)
% One stochastic exogenous state variable: z, an AR(1) process (in logs), idiosyncratic shock to labor productivity units
% Age: j
% Permanent types: only difference across ptypes is the value of parameter 

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=51; % Endogenous labour choice (fraction of time worked)
n_a=201; % Endogenous asset holdings
n_z=21; % Exogenous labor productivity units shock
N_j=Params.J; % Number of periods in finite horizon
N_i=5; % Number of permanent types

%% Set up the permanent types function
% We just need a function that take Params as an input and gives Params as
% an output. Inside the function we 
ParametrizeParamsFn=@(Parameters) LifeCycleModel49_PTypeFn(Parameters);
% Note, this will require N_i, so we need to store it in Params
Params.N_i=N_i;

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
% Fixed-effect in the labor productivity units
Params.logalpha_mean=0;
Params.logalpha_stddev=0.1;
% Following few lines just illustrate what PTypeDistParamNames function is doing
[alpha_grid,pi_alpha]=discretizeAR1_FarmerToda(Params.logalpha_mean,0,Params.logalpha_stddev,N_i);
pi_alpha=pi_alpha(1,:); % iid, just use first row
alpha_grid=exp(alpha_grid);
alpha_grid=alpha_grid*exp(Params.logalpha_mean)/sum(alpha_grid.*pi_alpha'); % normalize to mean exp(Params.logmean_stddev)
Params.alpha_i=alpha_grid;
PTypeDistParamNames={'alphadist'};
Params.alphadist=pi_alpha; % Must sum to one
% Note: this distribution is not relevant to solving the value function, but is needed for
% stationary distribition. It then gets encoded into the StationaryDist and
% so is not needed for things like life-cycle profiles.

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

% Use 'LifeCycleModel49_ReturnFn', only difference from 'LifeCycleModel45_ReturnFn' is that it includes alpha_i
ReturnFn=@(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,alpha_i,warmglow1,warmglow2,warmglow3,beta,sj) LifeCycleModel49_ReturnFn(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,alpha_i,warmglow1,warmglow2,warmglow3,beta,sj)

%% Now solve the value function iteration problem, just to check that things are working before we go to estimation
disp('Test ValueFnIter')
vfoptions.divideandconquer=1; % faster, requires problem is monotone
tic;
[V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,N_i, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, vfoptions);
toc

%% Initial distribution of agents at birth (j=1)
% Define how agents are at age j=1. We will set up a joint-normal distribution on assets and exogenous shocks.
% To be able to estimate the permanent type distribution we have to include permanent types in this initial distribution.

% Set initial assets to be joint-normal distribution with
Params.initassets_mean=1; % parameter for the mean of initial assets
Params.initassets_stddev=0.5; % parameter for the std dev of initial assets
Params.initz_mean=0; % parameter for the mean of initial z
Params.initz_stddev=Params.sigma_epsilon_z/(1-Params.rho_z); % parameter for the std dev of initial z

InitialDistCovarMatrix=[Params.initassets_stddev^2, 0; 0, Params.initz_stddev^2];

% So we need to put this joint-normal distribution onto our asset grid
jequaloneDist=MVNormal_ProbabilitiesOnGrid([a_grid; z_grid],[Params.initassets_mean; Params.initz_mean],InitialDistCovarMatrix,[n_a,n_z]); % note: first point in a_grid is zero, so have to add something tiny before taking log


%% We now compute the 'stationary distribution' of households
% Start with a mass of one at initial age, use the conditional survival
% probabilities sj to calculate the mass of those who survive to next
% period, repeat. Once done for all ages, normalize to one
Params.mewj=ones(1,Params.J); % Marginal distribution of households over age
for jj=2:length(Params.mewj)
    Params.mewj(jj)=Params.sj(jj-1)*Params.mewj(jj-1);
end
Params.mewj=Params.mewj./sum(Params.mewj); % Normalize to one
AgeWeightParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age
simoptions=struct(); % Use the default options
StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,N_i,pi_z,Params,simoptions);
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
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate,Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,z_grid,simoptions);

% For example
% AgeConditionalStats.earnings.Mean
% There are things other than Mean, but in our current deterministic model
% in which all agents are born identical the rest are meaningless.

%% Plot the life cycle profiles of fraction-of-time-worked, earnings, and assets

figure(1)
subplot(3,1,1); plot(1:1:Params.J,AgeConditionalStats.fractiontimeworked.Mean)
title('Life Cycle Profile (pre-calibration): Fraction Time Worked (h)')
subplot(3,1,2); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile (pre-calibration): Labor Earnings (w kappa_j h)')
subplot(3,1,3); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile (pre-calibration): Assets (a)')



%% Everything is working fine, time to turn to GMM Estimation of this model
% We will estimate the parameters of the normal distribution of the permanent types. 
% Params.logalpha_mean=0;
% Params.logalpha_stddev=0.1;

% As targets, we will use the age-conditional mean earnings.
% Since these targets come from our model, we know that the true parameter
% values are logalpha_mean=0 and logalpha_stddev=0.1.

% Obviously we will want to give a different initial guess for these parameters.
% The contents of Params (as passed as an input the the estimtion command below)
% are used as initial guesses for the parameters to be estimated, so we can
% just set some initial guesses as
Params.logalpha_mean=0.1;
Params.logalpha_stddev=0.05;

%% GMM setup, three simple steps
% First, just name all the model parameters we want to estimate
EstimParamNames={'logalpha_mean','logalpha_stddev'};
% EstimParamNames gives the names of all the model parameters that will be
% estimated, these parameters must all appear in Params, and the values in
% Params will be used as the initial values.
% All other parameters in Params will remain fixed.

% Std deviation must be positive, normally we might do this using
% estimoptions.constrainpositive
% but in this example, we already use exp(logalpha_stddev) which will be
% positive regardless of the value of logalpha_stddev, so no need here.

% Second, we need to say which model statistics we want to target
% We can target any model stat generated by the AllStats, and LifeCycleProfiles commands
% We set up a structure containing all of our targets
TargetMoments.AgeConditionalStats.earnings.Mean=AgeConditionalStats.earnings.Mean;
TargetMoments.AgeConditionalStats.earnings.Variance=AgeConditionalStats.earnings.Variance;
% Note: When setting up TargetMoments there are some rules you must follow
% There are two options TargetMomements.AgeConditionalStats and TargetMoments.AllStats (you can use both). 
% Within these you must follow the structure that you get when you run the commands
% AgeConditionalStats=LifeCycleProfiles_FHorz_Case1()
% and
% AllStats=EvalFnOnAgentDist_AggVars_FHorz_Case1()

% Note, targeting the retirement earnings would be silly, as the parameters
% are irrelevant to them. So let's drop them from what we want to estimate.
% This is easy, just set them to NaN and they will be ignored
TargetMoments.AgeConditionalStats.earnings.Mean(Params.Jr:end)=NaN; % drop the retirement periods from the estimation
TargetMoments.AgeConditionalStats.earnings.Variance(Params.Jr:end)=NaN; % drop the retirement periods from the estimation

% Third, we need a weighting matrix.
% We will just use the identity matrix, which is a silly choice, but easy.
% In Life-Cycle Model 48 we look at better ideas for how to choose the weighting matrix.
WeightingMatrix=eye(sum(~isnan([TargetMoments.AgeConditionalStats.earnings.Mean,TargetMoments.AgeConditionalStats.earnings.Variance])));

%% To be able to compute the confidence intervals for the estimated parameters, there is one other important input we need
% The variance-covariance matrix of the GMM moment conditions, which here
% just simplifies to the variance-covariance matrix of the 'data' moments.
% We will see get this from data in Life-Cycle Model 47, for now, here is one I prepared earlier.
CovarMatrixDataMoments=diag([[0.001, 0.001, 0.001, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002,...
    0.003, 0.003, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.005, 0.005, 0.005, 0.004, 0.004, 0.003, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001],[0.001, 0.001, 0.001, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002,...
    0.003, 0.003, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.005, 0.005, 0.005, 0.004, 0.004, 0.003, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001]/20]);


%% Done, now we just do the estimation

% We want a FnsToEvaluate which is just earnings (solely as this will be faster as it only includes what we actually need)
FnsToEvaluate_forGMM.earnings=FnsToEvaluate.earnings;

estimoptions.verbose=1; % give feedback
[EstimParams, EstimParamsConfInts, estsummary]=EstimateLifeCycleModel_PType_MethodOfMoments(EstimParamNames,TargetMoments, WeightingMatrix, CovarMatrixDataMoments, n_d,n_a,n_z,N_j,N_i, d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, jequaloneDist, AgeWeightParamNames,PTypeDistParamNames, ParametrizeParamsFn, FnsToEvaluate_forGMM, estimoptions, vfoptions, simoptions);
% EstimParams is the estimated parameter values
% EstimParamsConfInts are the 90% confidence intervals for the estimated parameter values
% estsummary is a structure containing various info on how the estimation went, plus some output useful for analysis

% Looking at
EstimParams
% we can see that the logalpha_mean is well estimated, being close to it's true value of 0 (and
% EstimParamsConfInts covers the true value). But logalpha_stddev is not
% well estimated (it is not way off the true value of 0.05, but not
% particularly close either) showing that it is not really identified by
% the moments we have chosen, in particular by the age-conditional variance
% of earnings (which makes sense, as it is tricky to differentiate the variance 
% in alpha_i from the variance in z, and the later is much larger).
% This acts as a lesson in thinking carefully about what moments you target.

% We discussed EstimParamsConfInts and estsummary in Life-Cycle Models 47 & 48.

%% Show some model output based on the estimated parameters
for pp=1:length(EstimParamNames)
    Params.(EstimParamNames{pp})=EstimParams.(EstimParamNames{pp});
end
[V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,N_i, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, vfoptions);
StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,N_i,pi_z,Params,simoptions);
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate,Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,z_grid,simoptions);
% Since our estimated parameters were essentially identical to the 'true'
% parameter values, obviously the following is going to look the same as
% the previous figure
figure(2)
subplot(3,1,1); plot(1:1:Params.J,AgeConditionalStats.fractiontimeworked.Mean)
title('Life Cycle Profile: Fraction Time Worked (h)')
subplot(3,1,2); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Labor Earnings (w kappa_j h)')
subplot(3,1,3); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Assets (a)')














