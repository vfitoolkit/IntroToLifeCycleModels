%% Life-Cycle Model 48: GMM Estimation of a Life-Cycle Model, various extras
% This is largely just redoing the GMM estimation in Life-Cycle Model 45,
% but showing how to use various estimoptions and looking at some of the
% other outputs.

% Essentially all changes are after the estimation, so at the bottom of
% this script from roughly line 314 on. The few exceptions are mentions just 
% below in square brackets [].

% First, instead of targeting the same moments, for half of them we target the log 
% of the moment. To do this we set estimoptions.logmoments to be a column 
% vector zeros of the length of the vector of moments to be estimated, but with ones 
% in the elements that correspond to the moments we wish to restrict to be positive.
% [see estimoptions.logmoments, around line 240]
% Note that the moments that we put into TargetMoments are not logged, the
% logs will be taken internally. But the covariancematrix must be already
% based on the log moments.

% Second, rather than targeting (log) age-conditional mean earnings we instead target 
% just the even ages, this just involves putting NaN in the target moments for any 
% that we want to ignore (in our case all the odd ages).
% [see TargetMoments, around line 250]

% Third, the default is reporting 90-percent confidence intervals for the estimated 
% parameters. We can change this using, e.g., estimoptions.confidenceintervals=95 to 
% get the 95-percent confidence intervals. If we are interested in standard deviations 
% of the estimated parameters, these can be found in estsummary.EstimParamsStdDev.

% Fourth, we look at whether the model parameters are locally-identified. A standard 
% approach is to look at the rank of J, and is reported in estsummary.localidentification.

% Fifth, we are interested in which moments are determining which parameters. 
% estsummary.sensitivitymatrix reports a sensitivity matrix; rows index the parameters, 
% column index the moments, and a large number (relative to parameter value) indicates that 
% this parameter is sensitive to this moment.

% Sixth, we might be interested in how sensitive our estimated parameters are to any 
% pre-calibrated parameters. Setting estimoptions.CalibParamsNames to, e.g., 
% estimoptions.CalibParamsNames={'w'} , means there will be a 
% estsummary.sensitivitytocalibrationmatrix which reports the sensitivity of estimated 
% parameters (rows) to the pre-calibrated parameters (columns).
% [see estimoptions.CalibParamsNames, around line 255]

% Seventh, before you estimate a model you need to think about which moments
% to target given the parameters you want to estimate. We are going to care
% about the derivates of the target moments with respect to the estimated
% parameters for two reasons, identification and the width of the confidince
% intervals. The width confidence intervals will roughly be the variance of
% the data moments divided by the derivative of the moment with respect to
% the parameter. So, conditional on the variance of the data moments, if we
% can find moments where the derivative with repsect to the parameter is
% large then the confidience intervals will be smaller. So we might be
% interest, before we collect data and start estimating, in looking at
% which moments have large derivatives with respect to which parameters.
% While in principle we care about the derivatives at the estimated
% parameter values, we don't have those before we start but it might still
% be useful to look at these derivatives for some initial parameter values.
% VFI Toolkit has a command 'EstimateLifeCycleModel_MomentDerivatives' that
% calculates all the derivatives for the lots of moments. So you can look
% through these to help think about what moments to target given the
% parameters you want to estimate; loosely, large derivatives mean these
% moments will be good for estimating these parameters (all else equal).

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

% Use 'LifeCycleModel45_ReturnFn' (it is just a renamed copy of the return fn used by Life-Cycle Model 9)
ReturnFn=@(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj) LifeCycleModel45_ReturnFn(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)

%% Now solve the value function iteration problem, just to check that things are working before we go to estimation
disp('Test ValueFnIter')
vfoptions.divideandconquer=1; % faster, requires problem is monotone
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

%% Initial distribution of agents at birth (j=1)
% Define how agents are at age j=1. We will set up a joint-normal
% distribution on assets and exogenous shocks.

% Set initial assets to be joint-normal distribution with
Params.initassets_mean=1; % parameter for the mean of initial assets
Params.initassets_stddev=0.5; % parameter for the std dev of initial assets
Params.initz_mean=0; % parameter for the mean of initial z
Params.initz_stddev=Params.sigma_epsilon_z/(1-Params.rho_z); % parameter for the std dev of initial z

InitialDistCovarMatrix=[Params.initassets_stddev^2, 0; 0, Params.initz_stddev^2];

% So we need to put this joint-normal distribution onto our asset grid
tic;
jequaloneDist=MVNormal_ProbabilitiesOnGrid([a_grid; z_grid],[Params.initassets_mean; Params.initz_mean],InitialDistCovarMatrix,[n_a,n_z]); % note: first point in a_grid is zero, so have to add something tiny before taking log
initdisttime=toc


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
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
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
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

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
% We will estimate three preference parameters.
% Preferences
% Params.sigma = 2; % Coeff of relative risk aversion (curvature of consumption)
% Params.eta = 1.5; % Curvature of leisure (This will end up being 1/Frisch elasty)
% Params.psi = 10; % Weight on leisure

% As targets, we will use the age-conditional mean earnings.
% Since these targets come from our model, we know that the true parameter
% values are sigma=2, eta=1.5, psi=10. 

% Obviously we will want to give a different initial guess for these parameters.
% The contents of Params (as passed as an input the the estimtion command below)
% are used as initial guesses for the parameters to be estimated, so we can
% just set some initial guesses as
Params.sigma=1.5;
Params.eta=1.1;
Params.psi=5;

%% GMM setup, three simple steps
% First, just name all the model parameters we want to estimate
EstimParamNames={'sigma','eta','psi'};
% EstimParamNames gives the names of all the model parameters that will be
% estimated, these parameters must all appear in Params, and the values in
% Params will be used as the initial values.
% All other parameters in Params will remain fixed.

% Second, we need to say which model statistics we want to target
% We can target any model stat generated by the AllStats, and LifeCycleProfiles commands
% We set up a structure containing all of our targets
TargetMoments.AgeConditionalStats.earnings.Mean=log(AgeConditionalStats.earnings.Mean); % We are going to target the log moment, which we have to say using estimoptions about 10 lines below this
% Note: When setting up TargetMoments there are some rules you must follow
% There are two options TargetMomements.AgeConditionalStats and TargetMoments.AllStats (you can use both). 
% Within these you must follow the structure that you get when you run the commands
% AgeConditionalStats=LifeCycleProfiles_FHorz_Case1()
% and
% AllStats=EvalFnOnAgentDist_AggVars_FHorz_Case1()

% We want to modify these targets.
% First, target the log of the moments (note that we already did log() in the target moments)
% The way to do log moments is just to put estimoptions.logmoments to have
% the same "AgeConditionalStats.earnings.Mean" as was used in the target moments.
estimoptions.logmoments.AgeConditionalStats.earnings.Mean=1;
% [Note: estimoptions.logmoments=1 is a simply way to log all target
% moments and would be the better choice in this example. But the above
% approach allows you to log some moments but not others by just putting
% the names of those moments you want to log]

% Second, only target the even ages (note, first model period represents
% age 20, so we are targeting the odd model periods). Do this using NaN for
% any targets we want to 'ignore'
TargetMoments.AgeConditionalStats.earnings.Mean(1:2:end)=NaN;
% No point to this, just to demonstrate how easy you can use NaN to omit certain moments from TargetMoments

% To evaluate the sensitivity of the estimated parameters to the
% pre-calibrated parameters, we need to give the names of the
% pre-calibrated parameters we are interested in
estimoptions.CalibParamsNames={'beta','pension'};

% Note, targeting the retirement earnings would be silly, as the parameters
% are irrelevant to them. So let's drop them from what we want to estimate.
% This is easy, just set them to NaN and they will be ignored
TargetMoments.AgeConditionalStats.earnings.Mean(Params.Jr:end)=NaN; % drop the retirement periods from the estimation

% Third, we need a weighting matrix.
% We will just use the identity matrix, which is a silly choice, but easy.
% In Life-Cycle Model 47 we look at better ideas for how to choose the weighting matrix.
WeightingMatrix=eye(sum(~isnan(TargetMoments.AgeConditionalStats.earnings.Mean)));

%% To be able to compute the confidence intervals for the estimated parameters, there is one other important input we need
% The variance-covariance matrix of the GMM moment conditions, which here
% just simplifies to the variance-covariance matrix of the 'data' moments.
% We will see get this from data in Life-Cycle Model 47, for now, here is one I prepared earlier.
CovarMatrixDataMoments=diag([0.001, 0.001, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002,...
    0.003, 0.004, 0.004, 0.004, 0.005, 0.004, 0.003, 0.003, 0.002, 0.002, 0.001]);


%% Done, now we just do the estimation
ParametrizeParamsFn=[]; % not something we are using in this example (will be used in Life-Cycle Model 47 and 49)

% We want a FnsToEvaluate which is just earnings (solely as this will be faster as it only includes what we actually need)
FnsToEvaluate_forGMM.earnings=FnsToEvaluate.earnings;

estimoptions.verbose=1; % give feedback
[EstimParams, EstimParamsConfInts, estsummary]=EstimateLifeCycleModel_MethodOfMoments(EstimParamNames,TargetMoments, WeightingMatrix, CovarMatrixDataMoments, n_d,n_a,n_z,N_j, d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, jequaloneDist, AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate_forGMM, estimoptions, vfoptions, simoptions);
% EstimParams is the estimated parameter values
% EstimParamsConfInts are the 90% confidence intervals for the estimated parameter values
% estsummary is a structure containing various info on how the estimation went, plus some output useful for analysis

% Looking at
EstimParams
% we can see that they are essentially the same as the 'true' values which were
% sigma=2, eta=1.5, psi=10


%% Show some model output based on the estimated parameters
for pp=1:length(EstimParamNames)
    Params.(EstimParamNames{pp})=EstimParams.(EstimParamNames{pp});
end
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);
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

%% Looking at other outputs
% If we want the value of the GMM objective function, (M_d-M_m)'W(M_d-M_m), it is in
estsummary.objectivefnval
% Actually, lots of info on the outputs can be found in estsummary.notes, for instance
estsummary.notes.objectivefnval
% tells us about what is in estsummary.objectivefnval

% Some important inputs and outputs are also recorded
estsummary.variousmatrices.W % the weighting matrix that we input
estsummary.variousmatrices.Omega % the covariance matrix of the data moments that we input
estsummary.variousmatrices.J % the Jacobian matrix of the derivatives of the model moments to the estimated parameters
estsummary.variousmatrices.Sigma % the covariance matrix of the estimated parameter vector
% J is calculated by taking derivatives (evaluated at the estimated parameter vector)
% Sigma is calculated as Sigma=(J'WJ)'^(-1) J'W'Omega W J (J'WJ)^(-1) [following the asymptotic theory for GMM estimators]

% The standard deviations of the estimated parameters are in
estsummary.EstimParamsStdDev
% These are just the square-root of the diagonal elements of Sigma

% One of the main outputs, as we saw in Life-Cycle Model 46, was the 90-percent 
% confidence intervals
EstimParamsConfInts
% These are calculated from the standard deviations, based on the fact that
% the estimated parameter vector is asymptotically normally distributed (so
% 90 percent confidence intervals are given by +-1.645*stddev).
% The 68,80,85,90,95,98,99 are all automatically computed, and can be found
% in, e.g., 
estsummary.confidenceintervals.confint68
estsummary.confidenceintervals.confint95
% They are calculated as +-[1,1.282,1.440,1.645, 1.96, 2.33, 2.575]*stddev.
% You can change which one is the main reported outcome in EstimParamsConfInts using
% estimoptions.confidenceintervals=90 (90 percent confidence intervals are
% the default)

%% Identification
% There is no point estimating the parameters using the target moments if
% the moments don't tell us anything about what the parameters should be;
% this is the issue of identification.
% The output includes info about local identification. There are two things
% that might go wrong in local identification. Changing a parameter might
% have no impact on the target moments, in which case those moments are
% obviously not going to tell us what the value of the parameter should be.
% Or, two parameters might have the exact same effects on the target
% moments, in which case we cannot tell them apart and so cannot pin them
% down.
% Both of these can be assessed by looking at the matrix J, the Jacobian 
% matrix of the derivatives of the model moments to the estimated
% parameters. If the matrix J is full rank then neither of the two issues
% described above is happening, and we conclude that the estimated
% parameters are locally identified.
% We find this in
estsummary.localidentification.rankJ % If this is greater than or equal to number of parameters, then locally identified
estsummary.localidentification.yesidentified % If this is 1, then rankJ is greater than or equal to number of parameters
% If you forget what these are about, there is an explanation in
estsummary.notes.localidentification

% Remember GMM estimation is about minimizing (M_d-M_m)'W(M_d-M_m). Local
% identification is about checking that we have found a local minimum.
% There is still a question of global identification, is this a global
% minimum. There is no easy way to answer this question (you could try lots
% of different initial guesses for the estimated parameter vector and if
% they all get to the same answer for the estimated parameter vector then
% the model is very likely globally identified).

%% Sensitivity
% We might be interested in thinking about how the moments pin down the
% parameters: which moments matter for which parameters? and how much would
% a change in the value of a given moment alter the value of a given
% parameter? This is question of sensistivity.

% We can look at the sensitivity matrix
estsummary.sensitivitymatrix
% The element in row i column j of this matrix is the sensitivity of
% estimated parameter i to target moment j.
% A specific example, using our current estimates. Element (2,1) of the
% sensitivity matrix is AAA  (I don't give the value, you can see it in code outputs), 
% and represents the sensitivity of the second estimated parameter, eta, to 
% the first target moment, mean earnings at age 20. It is telling us that if 
% mean earnings at age 20 increased by x, then the estimated eta parameter 
% would increase by y=AAA*x.
% This sensitivity matrix is from Andrews, Gentzkow & Shapiro (2017) - Measuring the Sensitivity of Parameter Estimates to Estimation Moments
% (they call it Lambda). And the formula is
% SensitivityMatrix=(-(J'*W*J)^(-1))*(J'*W)
% The interpretation is in estsummary.notes.sensitivitymatrix so as to be handy


% In practice, it is common to split all the model parameters into two sets. 
% We calibrate the model parameters in the first set. And then after this we
% estimate the model parameters in the second set. A natural question is
% how sensitive the values of the estimated parameters are to the
% pre-calibrated parameters.
% To look at this we first have to say the names of the pre-calibrated
% parameters that we are interested in 
% (we did this above, estimoptions.CalibParamsNames={'beta','pension'}; )
% We can then look at the sensitivity matrix
estsummary.sensitivitytocalibrationmatrix
% The element in row i column j of this matrix is the sensitivity of
% estimated parameter i to pre-calibrated parameter j.
% A specific example, using our current estimates. Element (2,1) of the
% sensitivity matrix is AAA (I don't give the value, you can see it in code outputs), 
% and represents the sensitivity of the second estimated parameter, eta, to 
% the first pre-calibrated parameter, beta. It is telling us that if beta 
% increased by x, then the estimated eta parameter would increase by y=AAA*x.
% This sensitivity matrix is from Jorgensen (2023) - Sensitivity to Calibrated Parameters
% And the formula is
% SensitivityToCalibrationMatrix=SensitivityMatrix*J
% The interpretation is in estsummary.notes.sensitivitytocalibrationmatrix so as to be handy

% As for identification, this is all about local sensitivity (how small
% changes in the target moments/calibrated parameters will change the value
% of the estimated parameters). We could also be interested in the question
% of global sensitivity. Again, global is more difficult to evaluate than
% local and we have nothing to say about it here.


%% Derivatives of moments with respect to parameters
% Before you estimate a model you need to think about which moments
% to target given the parameters you want to estimate. We are going to care
% about the derivates of the target moments with respect to the estimated
% parameters for two reasons, identification and the width of the confidince
% intervals. The width confidence intervals will roughly be the variance of
% the data moments divided by the derivative of the moment with respect to
% the parameter. So, conditional on the variance of the data moments, if we
% can find moments where the derivative with repsect to the parameter is
% large then the confidience intervals will be smaller. So we might be
% interest, before we collect data and start estimating, in looking at
% which moments have large derivatives with respect to which parameters.
% While in principle we care about the derivatives at the estimated
% parameter values, we don't have those before we start but it might still
% be useful to look at these derivatives for some initial parameter values.
% VFI Toolkit has a command 'EstimateLifeCycleModel_MomentDerivatives' that
% calculates all the derivatives for the lots of moments. So you can look
% through these to help think about what moments to target given the
% parameters you want to estimate; loosely, large derivatives mean these
% moments will be good for estimating these parameters (all else equal).

estimoptions.logmoments=0; % For EstimateLifeCycleModel_MomentDerivatives, estimoptions.logmoments must be scalar either 0 or 1
[MomentDerivatives,SortedMomentDerivatives,momentderivsummary]=EstimateLifeCycleModel_MomentDerivatives(EstimParamNames, n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, estimoptions, vfoptions,simoptions);
% Compute derivatives of model moments with respect to the 'estimated parameters'
%
% dM_m(theta)/dtheta
%
% M_m are all the moments the toolkit does with 'AllStats' and 'LifeCycleProfiles'
% theta is a vector of parameters.

% Here we cheat, as we have the estimated parameter values in Params.
% Obviously in practice you would just have to do this with some reasonable
% initial guesses in Params.

%% Final note
estimoptions.skipestimation=1;
% can be used to recalculate all the outputs of
% EstimateLifeCycleModel_MethodOfMoments()
% without repeating the estimation (uses whatever the inital parameter vector is when called). 
% For example, after you complete the estimation you could decide you want
% to perform the sensitivity with respect to the pre-calibrated parameters.
% In which case you can set estimoptions.CalibParamsNames and estimoptions.skipestimation=1
% and then call EstimateLifeCycleModel_MethodOfMoments() to get the sensitivity to 
% the calibrated parameters without having to wait for the entire
% estimation again.
% Could also be used, e.g., to check if using larger grids changes the Jacobian 
% matrix (J) (and hence the standard deviations and confidence intervals for the
% estimated parameter vector).



