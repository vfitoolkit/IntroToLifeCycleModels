%% Life-Cycle Model 40: Two Endogenous States (Housing)
% Two endogenous states: a (financial assets), h (housing)
% One exogenous state: z (AR(1) on labor productivity units)
%
% The housing setup follows Chen (2010): agents can rent or own. Owners
% receive housing services equal to the size of their house, hprime.
% Renters purchase housing services d at price p. Transactions costs are
% paid when h changes. Owners can borrow with their house as collateral,
% subject to a downpayment requirement (gamma).
%
% Setting up two endogenous states is easy enough with, e.g., n_a=[301,21]; and
% then using a stacked column vector for the a_grid.
%
% For ReturnFn and FnsToEvaluate, two endogenous states are set up as
% (a1prime,a2prime,a1,a2,z,...). If the model had decision variables, those
% would come before, so (d,a1prime,...).
%
% We will use vfoptions.divideandconquer and vfoptions.gridinterplayer, both
% of which are hardcoded to be applied to the first of the two endogenous
% states. Given that one of our endogenous states is housing, we want to
% put fewer points on housing (just limit it to 21 different values) and
% so we should put housing as the second endogenous state so that we can
% use divide-and-conquer and grid interpolation layer on assets, which will
% have way more grid points.
%
% Note that the decision variable d, the amount of housing services
% purchased by a renter, can be solved for analytically and so does not
% need to be included in the code as a decision variable (instead the
% analytic solution is just included in the return function as a formula).
% See http://discourse.vfitoolkit.com/t/models-with-housing-a-simplification-to-handle-renters-splitting-cspend-analytically-into-consumption-c-and-housing-services-d/653
% for the derivation.

%% How does VFI Toolkit think about this?
%
% No decision variable. Can set n_d=0, d_grid=[]
% Two endogenous state variables: a, financial assets; h, housing
% One stochastic exogenous state variable: z, an AR(1) process (in logs), idiosyncratic shock to labor productivity units
% Age: j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=0; % No decision variables
n_a=[301,21]; % Endogenous asset holdings, housing
n_z=21; % Exogenous labor productivity units shock
N_j=Params.J; % Number of periods in finite horizon

%% Parameters

% Discount rate
Params.beta = 0.96;
% Preferences
Params.sigma = 2; % Coeff of relative risk aversion (curvature of utility)
Params.theta = 0.8; % share of nondurable consumption in utility (vs housing services)
Params.upsilon = 0; % CES parameter on (c,d); =0 implies unit elasticity of substitution (Cobb-Douglas)

% Prices
Params.w=1; % Wage
Params.r=0.05; % Interest rate (0.05 is 5%)
Params.p=0.07; % Rental price of housing services (per unit of housing, per period)

% Housing
Params.delta_o=0.013; % Depreciation rate for owner-occupied housing
Params.phi=0.05; % Transactions cost for changing housing (fraction of current house value)
Params.gamma=0.2; % Downpayment ratio (collateral constraint: aprime>=-(1-gamma)*hprime)

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
% US data, taken from "National Vital Statistics Report, volume 58, number 10, March 2010."
% Conditional death probabilities (first column (qx) of Table 1, Total Population)
Params.dj=[0.006879, 0.000463, 0.000307, 0.000220, 0.000184, 0.000172, 0.000160, 0.000149, 0.000133, 0.000114, 0.000100, 0.000105, 0.000143, 0.000221, 0.000329, 0.000449, 0.000563, 0.000667, 0.000753, 0.000823,...
    0.000894, 0.000962, 0.001005, 0.001016, 0.001003, 0.000983, 0.000967, 0.000960, 0.000970, 0.000994, 0.001027, 0.001065, 0.001115, 0.001154, 0.001209, 0.001271, 0.001351, 0.001460, 0.001603, 0.001769, 0.001943, 0.002120, 0.002311, 0.002520, 0.002747, 0.002989, 0.003242, 0.003512, 0.003803, 0.004118, 0.004464, 0.004837, 0.005217, 0.005591, 0.005963, 0.006346, 0.006768, 0.007261, 0.007866, 0.008596, 0.009473, 0.010450, 0.011456, 0.012407, 0.013320, 0.014299, 0.015323,...
    0.016558, 0.018029, 0.019723, 0.021607, 0.023723, 0.026143, 0.028892, 0.031988, 0.035476, 0.039238, 0.043382, 0.047941, 0.052953, 0.058457, 0.064494,...
    0.071107, 0.078342, 0.086244, 0.094861, 0.104242, 0.114432, 0.125479, 0.137427, 0.150317, 0.164187, 0.179066, 0.194979, 0.211941, 0.229957, 0.249020, 0.269112, 0.290198, 0.312231, 1.000000];
% dj covers Ages 0 to 100
Params.sj=1-Params.dj(21:101); % Conditional survival probabilities
Params.sj(end)=0; % In the present model the last period (j=J) value of sj is actually irrelevant

% Warm glow of bequest
Params.wg1=0.3; % (relative) importance of bequests
Params.wg2=3; % degree to which bequests are a luxury good (>=1; =1 would be a normal good)
Params.wg3=Params.sigma; % By using the same curvature as the utility of consumption it makes it much easier to guess appropriate parameter values for the warm glow

%% Grids
% Housing grid, from 0 to maxh. hprime=0 means renter; hprime>0 means owner.
minh=0;
maxh=5;
h_grid=minh+(maxh-minh)*linspace(0,1,n_a(2))'.^2; % ^2 puts more points near minh

% Asset grid. Includes a negative portion to allow mortgage borrowing.
% The minimum possible value of assets is -(1-gamma)*maxh (the maximum mortgage).
minassets=-(1-Params.gamma)*maxh;
maxassets=10;
% Negative part of the grid: evenly spaced from minassets up to 0.
n_a_neg=round(0.1*n_a(1)); % roughly 10% of points in the negative range
asset_grid_neg=linspace(minassets,0,n_a_neg)';
% Positive part of the grid (more points nearer 0, where the value fn is most curved).
asset_grid_pos=maxassets*linspace(0,1,n_a(1)-n_a_neg+1)'.^3;
% Note: both contain zero, so omit it from asset_grid_neg before stacking
asset_grid=[asset_grid_neg(1:end-1); asset_grid_pos];

% Stacked column vector for the two endogenous states
a_grid=[asset_grid; h_grid];

% AR(1) process for z (labor productivity units)
[z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_epsilon_z,n_z);
z_grid=exp(z_grid); % Take exponential of the grid
[mean_z,~,~,~]=MarkovChainMoments(z_grid,pi_z); % Mean of the grid so as can normalise it
z_grid=z_grid./mean_z; % Normalise the grid on z (so that the mean of z is 1)

d_grid=[]; % No decision variables

%% Now, create the return function
DiscountFactorParamNames={'beta','sj'};

ReturnFn=@(aprime,hprime,a,h,z,w,r,p,sigma,theta,upsilon,gamma,phi,delta_o,agej,Jr,pension,kappa_j,wg1,wg2,wg3,beta,sj) ...
    LifeCycleModel40_ReturnFn(aprime,hprime,a,h,z,w,r,p,sigma,theta,upsilon,gamma,phi,delta_o,agej,Jr,pension,kappa_j,wg1,wg2,wg3,beta,sj);
% (aprime,hprime,a,h,z,...): two next-period endogenous states, then two current
% endogenous states, then the exogenous state z, then parameters.

%% Solve the value function
% With two standard endogenous states, divide-and-conquer and the grid interpolation
% layer are applied (only) to the first endogenous state. That's why we set
% up housing as the second endogenous state (see comment at top of file).
vfoptions.divideandconquer=1; % turn on divide-and-conquer
vfoptions.gridinterplayer=1; % turn on grid interpolation layer
vfoptions.ngridinterp=20; % 20 evenly-spaced points between each pair of consecutive grid points on the first endogenous state
simoptions.gridinterplayer=vfoptions.gridinterplayer; % grid interpolation layer must also be set in simoptions (it changes Policy size/interpretation)
simoptions.ngridinterp=vfoptions.ngridinterp;

disp('Solve for Value fn and Policy fn using ValueFnIter command')
tic;
[V,Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,pi_z,ReturnFn,Params,DiscountFactorParamNames,[],vfoptions);
vftime=toc

%% Initial distribution of agents at birth (j=1)
% All agents are born with zero assets and zero housing (so they are renters at j=1).
jequaloneDist=zeros([n_a,n_z],'gpuArray');
[~,zeroassetindex]=min(abs(asset_grid));
jequaloneDist(zeroassetindex,1,floor((n_z+1)/2))=1; % zero assets, zero housing, median z

%% Stationary distribution of households
Params.mewj=ones(1,Params.J); % Marginal distribution of households over age
for jj=2:length(Params.mewj)
    Params.mewj(jj)=Params.sj(jj-1)*Params.mewj(jj-1);
end
Params.mewj=Params.mewj./sum(Params.mewj); % Normalize to one
AgeWeightsParamNames={'mewj'};

StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% First inputs must be (aprime,hprime,a,h,z), then any relevant parameters.
FnsToEvaluate.earnings=@(aprime,hprime,a,h,z,w,kappa_j) w*kappa_j*z; % labor earnings (zero in retirement since kappa_j=0)
FnsToEvaluate.assets=@(aprime,hprime,a,h,z) a; % financial assets
FnsToEvaluate.housing=@(aprime,hprime,a,h,z) h; % current housing stock
FnsToEvaluate.homeownership=@(aprime,hprime,a,h,z) (hprime>0); % =1 if owner next period, =0 if renter
FnsToEvaluate.totalwealth=@(aprime,hprime,a,h,z) a+h; % financial assets plus housing
FnsToEvaluate.loantovalue=@(aprime,hprime,a,h,z) (aprime<0)*(hprime>0)*abs(aprime)/max(hprime,eps); % LTV ratio (only for borrowers)
FnsToEvaluate.consumption=@(aprime,hprime,a,h,z,w,r,p,theta,upsilon,phi,delta_o,agej,Jr,pension,kappa_j) ...
    LifeCycleModel40_ConsumptionFn(aprime,hprime,a,h,z,w,r,p,theta,upsilon,phi,delta_o,agej,Jr,pension,kappa_j);
FnsToEvaluate.housingservices=@(aprime,hprime,a,h,z,w,r,p,theta,upsilon,phi,delta_o,agej,Jr,pension,kappa_j) ...
    LifeCycleModel40_HousingServicesFn(aprime,hprime,a,h,z,w,r,p,theta,upsilon,phi,delta_o,agej,Jr,pension,kappa_j);

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

%% Some aggregate statistics, with stats conditional on being a homeowner
simoptions.conditionalrestrictions.Homeowners=@(aprime,hprime,a,h,z) (hprime>0);
AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

fprintf('Quantitative properties of the benchmark economy \n')
fprintf('Homeownership rate is %2.1f%% \n',100*AllStats.homeownership.Mean)
fprintf('Gini for total wealth is %1.2f \n',AllStats.totalwealth.Gini)
fprintf('Gini for financial wealth is %1.2f \n',AllStats.assets.Gini)
fprintf('Gini for housing is %1.2f \n',AllStats.housing.Gini)
fprintf('Mean loan-to-value ratio (for borrowing homeowners) is %2.1f%% \n',100*AllStats.Homeowners.loantovalue.Mean)

%% Plot the life-cycle profiles
agevec=Params.agejshifter+(1:1:Params.J);

figure(1)
subplot(3,1,1); plot(agevec,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Labor Earnings (w \kappa_j z)')
subplot(3,1,2); plot(agevec,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Financial Assets (a)')
subplot(3,1,3); plot(agevec,AgeConditionalStats.housing.Mean)
title('Life Cycle Profile: Housing (h)')

figure(2)
plot(agevec,AgeConditionalStats.consumption.Mean)
hold on
plot(agevec,AgeConditionalStats.housingservices.Mean)
plot(agevec,AgeConditionalStats.earnings.Mean)
hold off
legend('consumption','housing services','earnings')
title('Consumption, Housing Services and Earnings')

figure(3)
plot(agevec,AgeConditionalStats.homeownership.Mean)
title('Home-ownership rate (by age)')
ylim([0,1])
