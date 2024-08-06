%% Life-Cycle Model 47: GMM Estimation of a Life-Cycle Model
% Model is same as we had in Life-Cycle Model 45.
% Main thing we do here is use data to get data moments (the target
% moments), the covariance matrix of the data moments (needed for
% calculating the covariance matrix of the estimated parameter vector), and
% also look at how data can be used for the weighting matrix, as well as
% comparing some different options in terms of the weighting matrix.

% We will use a profile for age-conditional mean earnings as our target.
% We estimate the parameter kappa_j (the determinitistic component of earnings, which is age-dependent)
% Note, this is not simple because there is endogenous labor.
paramerizeKappaj=1; % Can be set to 0 or 1
% =1: We parameterize kappa_j (as a fifth order polynomial) and then estimate 
%     the co-efficients of the parametrization (six parameters of a third-order 
%     polynomial).
% =0: we estimate kappa_j as 45 different numbers (the age-conditional values, for each working age)
% Note that in practice it is standard to parametrize age-dependent
% variables, like kappa_j, as polynomials. We set this up as an option so you can
% see how the toolkit can handle both approaches.

% We use the publically available PSID data to estimate the age-conditional
% mean of earnings (you will need to download the data before you run this
% command, as explained a few lines below). When we do the data work we
% show how to do time vs cohort fixed effects (and discuss the issue
% briefly). 
% We then use the same data so show how to estimate the covariance matrix
% of the data moments. (What the theory calls Omega.)

% We then perform the estimation three times, using different weighting matrices.
% As long as the weighting matrix is positive semi-definite, the GMM
% estimator will be consistent and asymptotically normal.
% First we use W=I, the identity matrix. This is just demonstrated as an easy choice.
% Second we use W has on the diagonal the inverse of the diagonal elements of Omega (the
%    covariance matrix of the data moments), with zeros on the off-diagonal elements
% Third we use W equals the inverse of the Omega. This is efficient GMM
%
% The first W=I has the problem of being 'scale dependent', if we measure
% the earnings targets in dollars of thousands of dollars we will get
% different estimated parameter vectors.
% The second W is an example of a weighting matrix that avoids scale
% dependence. And is a popular choice in the literature.
% The third is called 'efficient' because it gives the smallest values for
% the covariance matrix of the estimated parameter vector. But it can
% suffer from being biased in small samples.

% We will start with the data work. 
figure_c=0; % a counter for the figures
% This all hidden inside 'LifeCycleModel47_DataWork'.
% Right-click on the following line and select 'Open'.
LifeCycleModel47_DataWork
% The first few lines of LifeCycleModel47_DataWork explain how to download
% the PSID data that is used (PSID=Panel Survey of Income Dynamics, it is
% publicly available US panel-data of households.)
% You will need to register an account on the PSID website, but this only
% takes a few minutes.

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

if paramerizeKappaj==1
    % We want to parametrize (log) kappa_j as a fifth-order polynomial, and then estimate the coefficients of this polynomial. We do this by using
    % ParametrizeParamsFn which takes the parameter structure as input and gives the parameter structure as output (so it will take our coefficients
    % of the polynomial inputs and give the kappa_j vector output; all being stored in the parameters structure).
    Params.kappa_j_c0=log(2); % Note, these values will be the initial guess, which is a very naive initial guess
    Params.kappa_j_c1=0;
    Params.kappa_j_c2=0;
    Params.kappa_j_c3=0;
    Params.kappa_j_c4=0;
    Params.kappa_j_c5=0;
    ParametrizeParamsFn=@(Params) LifeCycleModel47_ParametrizeKappajFn(Params);
    % LifeCycleModel47_ParametrizeKappajFn creates Params.kappa_j based on the
    % values of [kappa_j_c0, kappa_j_c1,...,kappa_j_c5] (evalutes the fifth-order polynomial on kappa_j)
    Params=ParametrizeParamsFn(Params);
else
    % Age-dependent labor productivity units
    Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
    ParametrizeParamsFn=[];
end

% Params.kappa_j=Params.kappa_j*(10/mean(Params.kappa_j)); % make it mean 10, so as when I replace it with data the psi value is still reasonable
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

% From the data we got the mean-earnings life-cycle profile and the covar matrix 
% MeanEarningsProfile_CFE
% CovarMatrixDataMoments_CFE
% (estimated with cohort-fixed-effects)
% Note: Units for earnings are 'thousands of 1969 dollars'
% Note: our data is working-age earnings, so we will fill in zeros for
% retirement age earnings.

% We target the mean-earnings life-cycle profile
% We will estimate the kappa_j (age-dependent labor productivity)

% Note that because labor is endogenous this is far from a trivial exercise.

if paramerizeKappaj==1    
    % As an initial guess, we prentend labor supply is 0.5 at all ages
    % I just use the mean as the constant term, and ignore all the other polynomial coefficients
    Params.kappa_j_c0=log(mean(MeanEarningsProfile_CFE'/0.5)); % Note, these values will be the initial guess, which is a very naive initial guess
    % note that earnings=w*kappa_j*h*z, and w=1, E[z]=1
    Params.kappa_j_c1=0;
    Params.kappa_j_c2=0;
    Params.kappa_j_c3=0;
    Params.kappa_j_c4=0;
    Params.kappa_j_c5=0;
else
    % As an initial guess, we prentend labor supply is 0.5 at all ages
    Params.kappa_j=[MeanEarningsProfile_CFE'/0.5,zeros(1,N_j-Params.Jr+1)];
    % note that earnings=w*kappa_j*h*z, and w=1, E[z]=1
end


%% GMM setup, three simple steps
% First, just name all the model parameters we want to estimate
if paramerizeKappaj==1
    EstimParamNames={'kappa_j_c0','kappa_j_c1','kappa_j_c2','kappa_j_c3','kappa_j_c4','kappa_j_c5'};
elseif paramerizeKappaj==0
    EstimParamNames={'kappa_j'};
end
% EstimParamNames gives the names of all the model parameters that will be
% estimated, these parameters must all appear in Params, and the values in
% Params will be used as the initial values.
% All other parameters in Params will remain fixed.

if paramerizeKappaj==1
    % Setting negative kappa_j in retirement, and then choosing to work zero hours, will give zero earnings in retirement (which is what the
    % target moments will have), but it is silly so we constrain kappa_j to be positive to avoid this
    % We used a log polynomial in the ParametrizeParamsFn and this is constraining kappa_j to be positive.
elseif paramerizeKappaj==0
    % Setting negative kappa_j in retirement, and then choosing to work zero hours, will give zero earnings in retirement (which is what the
    % target moments will have), but it is silly so we constrain kappa_j to be positive to avoid this
    estimoptions.constrainpositive={'kappa_j'};

    % We don't want to estimate kappa_j in retirement (it is undefined, but we
    % want to give it zero values). We set this up as
    estimoptions.omitestimparam.kappa_j=[nan(Params.Jr-1,1); zeros(N_j-Params.Jr+1,1)];
end

% Second, we need to say which model statistics we want to target
% We can target any model stat generated by the AllStats, and LifeCycleProfiles commands
% We set up a structure containing all of our targets
TargetMoments.AgeConditionalStats.earnings.Mean=[MeanEarningsProfile_CFE; nan(N_j-Params.Jr+1,1)]; % Note: NaN in retirement ages to omit them
% Note: When setting up TargetMoments there are some rules you must follow
% There are two options TargetMomements.AgeConditionalStats and TargetMoments.AllStats (you can use both). 
% Within these you must follow the structure that you get when you run the commands
% AgeConditionalStats=LifeCycleProfiles_FHorz_Case1()
% and
% AllStats=EvalFnOnAgentDist_AggVars_FHorz_Case1()

%% Third, we need a weighting matrix.
% We start by using the identity matrix.
% After we estimate, we then look below at two better choices for the weighting matrix.
WeightingMatrix=eye(sum(~isnan(TargetMoments.AgeConditionalStats.earnings.Mean))); 

%% To be able to compute the confidence intervals for the estimated parameters, there is one other important input we need
% The variance-covariance matrix of the GMM moment conditions, which here
% just simplifies to the variance-covariance matrix of the 'data' moments.
% We estimated this from data as CovarMatrixDataMoments_CFE
CovarMatrixDataMoments=CovarMatrixDataMoments_CFE;


%% Done, now we just do the estimation

% We want a FnsToEvaluate which is just earnings (solely as this will be faster as it only includes what we actually need)
FnsToEvaluate_forGMM.earnings=FnsToEvaluate.earnings;

estimoptions.verbose=1; % give feedback
[EstimParams1, EstimParamsConfInts1, estsummary1]=EstimateLifeCycleModel_MethodOfMoments(EstimParamNames,TargetMoments, WeightingMatrix, CovarMatrixDataMoments, n_d,n_a,n_z,N_j, d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, jequaloneDist, AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate_forGMM, estimoptions, vfoptions, simoptions);
% EstimParams is the estimated parameter values
% EstimParamsConfInts are the 90% confidence intervals for the estimated parameter values
% estsummary is a structure containing various info on how the estimation went, plus some output useful for analysis

% Looking at
EstimParams1
% we can see the estimated kappa_j


%% Show some model output based on the estimated parameters
for pp=1:length(EstimParamNames)
    Params.(EstimParamNames{pp})=EstimParams1.(EstimParamNames{pp});
end
if paramerizeKappaj==1
    Params=ParametrizeParamsFn(Params); % Update kappa_j based on the estimated coefficients
    Params.kappa_j(Params.Jr:end)=0; % clean this up (these numbers were essentially ignored by the estimation, as they do nothing to the model ---look closely at the ReturnFn and this will be obvious--- but I just like making them zero as is easier for me to look at and understand)
end
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);
% Notice how the earnings profile looks like that from the data
figure(2)
subplot(3,1,1); plot(1:1:Params.J,AgeConditionalStats.fractiontimeworked.Mean)
title('Life Cycle Profile: Fraction Time Worked (h)')
subplot(3,1,2); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Labor Earnings (w kappa_j h)')
subplot(3,1,3); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Assets (a)')

%% Plot model vs data
figure(3)
plot(1:1:Params.J,AgeConditionalStats.earnings.Mean, 1:1:Params.Jr-1, MeanEarningsProfile_CFE)
title('Age-conditional mean earnings: Data vs model')
legend('model','PSID data')

%% So far, so good. We used data and estimated the model.

%% Now, let's think about options for the weighting matrix
% We have used W=I, the identity matrix. This works, but is a very naive choice.

% Using W=Omega^(-1) gives us "efficient GMM"
% Efficient GMM is explained in the Appendix of the Intro to Life-Cycle Models about GMM estimation
% https://raw.githubusercontent.com/vfitoolkit/IntroToLifeCycleModels/main/Documentation/LifeCycleModels.pdf

%% Efficient GMM
WeightingMatrix=CovarMatrixDataMoments_CFE^(-1);
[EstimParams2, EstimParamsConfInts2, estsummary2]=EstimateLifeCycleModel_MethodOfMoments(EstimParamNames,TargetMoments, WeightingMatrix, CovarMatrixDataMoments, n_d,n_a,n_z,N_j, d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, jequaloneDist, AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate_forGMM, estimoptions, vfoptions, simoptions);


%% Another option is W=diag(Omega)^(-1) [zeros on the off-diagonals]

% The efficient GMM can be biased in small samples (because of correlation
% between our estimate of Omega, and our estimates of M^d, the data moments).

% The identity matrix is scale-dependent (if we change our moments from
% being measured in thousands of dollars to being measured in dollars, then
% the estimates change).

% So another option is to use
WeightingMatrix=diag(diag(CovarMatrixDataMoments_CFE.^(-1)));
% This eliminates the issue of scale-dependence (especially important if,
% e.g., one moment is measured in dollars and another moment is measured in
% hours). It will be less efficient than W=Omega^(-1), but is more likely
% to be robust in small samples (less likely to suffer from bias).
[EstimParams3, EstimParamsConfInts3, estsummary3]=EstimateLifeCycleModel_MethodOfMoments(EstimParamNames,TargetMoments, WeightingMatrix, CovarMatrixDataMoments, n_d,n_a,n_z,N_j, d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, jequaloneDist, AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate_forGMM, estimoptions, vfoptions, simoptions);

%% GMM Theory tells us that any positive semi-definite weighting matrix W will converge to the true estimated parameters.
% So the decision of which weighting matrix to choose is about balancing robustness against efficiency. 
% As can be seen above, trying, e.g., both W=Omega^(-1) and W=diag(Omega)^(-1) is easy enough.

% Notice that all three give similar parameter estimates
EstimParams1 % W=I
EstimParams2 % Efficient GMM, W=Omega^(-1)
EstimParams3 % W=diag(Omega)^(-1)
% And that the efficient GMM has the smallest confidence intervals (this is what efficient means)
EstimParamsConfInts1 % W=I
EstimParamsConfInts2 % Efficient GMM, W=Omega^(-1)
EstimParamsConfInts3 % W=diag(Omega)^(-1)

save LCM47.mat



