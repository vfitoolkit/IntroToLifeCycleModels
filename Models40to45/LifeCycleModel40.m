%% Life-Cycle Model 40: two endogenous states (Housing)
%
% Two endogenous states: a (financial assets), h (housing)
% One exogenous state: z (stochastic earnings)
%
% Setting up two endogenous states is easy enough with, e.g., n_a=[301,20]; and
% then using a stacked column vector for the a_grid.
%
% For ReturnFn and FnsToEvaluate, two endogenous states are set up as
% (a1prime,a2prime,a1,a2,z,...)
% If the model had decision variables, those would come before, so (d,a1prime,...)
%
% We will use vfoptions.divideandconquer and vfoptions.gridinterplayer, both
% of which are hardcoded to be applied to the first of the two endogenous
% states. Given that one of our endogenous states is housing, we want to
% put fewer points on housing (just limit it to 20 different values) and
% so we should put housing as the second endogenous state so that we can
% use divide-and-conquer and grid interpolation layer on assets, which will
% have way more grid points.
%
% Note that the decision variable d, the amount of housing-services
% purchased by a renter, can be solved for analytically and so does not
% need to be included in the code as a decision variable (instead the
% analytic solution is just included in the return function as a formula)


%% 
n_d=0; % no decision variables
n_a=[301,20]; % Assets, Housing 
n_z=7; % AR(1) process on earnings

N_j=81; % total number of periods
Params.agejshifter=19; % starting age 20
% So model is annual for ages 20 to 100
Params.J=N_j;

%% Parameters

% Discount factor
Params.beta=0.9622;

% Preferences
Params.sigma=2; % curvature of utility
Params.upsilon=0; % =0 implies a unit elasticity of substitution between the two types of consumption (consumption good and housing services)
Params.theta=0.8954; % share of nondurable consumption in the utility function

% Demographics
Params.Jr=46; % retire at age 65
Params.agej=1:1:Params.J; % agej is the period
Params.n=0.01; % Population growth rate

% Production function
Params.alpha=0.2743; % share of capital in Cobb-Douglas production fn
% Depreciation rates
Params.delta_r=0.0254; % annual depreciation rate for rental housing
Params.delta_o=0.013; % annual depreciation rate for owner-occupied housing
Params.delta_k=0.0951; % annual depreciation rate for physical capital (financial assets)

% Housing
Params.phi=0.05; % transaction cost for changing amount of housing
Params.gamma=0.2; % downpayment ratio (when buying a house)

% Earnings
Params.rho_z=0.96; % autocorrelation coeff of AR(1) on earnings
Params.sigma_z_epsilon=sqrt(0.045); % standard deviation of innovations to the AR(1) on earnings
Params.sigma_z1=sqrt(0.38); % standard deviation of initial earnings (at age 20, period 1)
% Note: asymptotic std dev of z is sqrt((0.045^2)/(1-0.96^2))=0.1607

Params.r=0.064; % Interest rate (rate of return on assets)

% p and w are both determined in general eqm. But since both are just
% simple functions of other general eqm parameters, we can just hard-code
% them (inside return function and elsewhere).
Params.p=(Params.r+Params.delta_r)/(1+Params.r); % price of housing
Params.w=(1-Params.alpha)*((Params.r+Params.delta_k)/Params.alpha)^(Params.alpha/(Params.alpha-1)); % wage

% Because labor supply is exogenous you can actually figure out of b is without having to
% solve general eqm (point 4 in Definition 1 of his Appendix A.1).
Params.b=0.1;
% This is an initial guess, and then we set b to clear the general eqm
% below. It is just below the first creation of AllStats.


% Taxes
Params.tau_p=0.107;

% Deterministic income profiles
Params.kappaj=[1.0000, 1.0719, 1.1438, 1.2158, 1.2842, 1.3527, 1.4212, 1.4897, 1.5582, 1.6267, 1.6952, 1.7217, 1.7438, 1.7748, 1.8014, 1.8279, 1.8545, 1.8810, 1.9075, 1.9341, 1.9606, 1.9623, 1.9640, 1.9658, 1.9675, 1.9692, 1.9709, 1.9726, 1.9743, 1.9760, 1.9777, 1.9700, 1.9623, 1.9546, 1.9469, 1.9392, 1.9315, 1.9238, 1.9161, 1.9084, 1.9007, 1.8354, 1.7701, 1.7048, 1.6396];
Params.kappaj=[Params.kappaj,zeros(1,Params.J-Params.Jr+1)]; % zeros for retirement

% Survival probabilities
% F. C. Bell and M. L. Miller (2005), Life Tables for the United States Social Security Area 1900-2100, Actuarial Study No. 120, Office of the Chief Actuary
% http://www.socialsecurity.gov/oact/NOTES/s2000s.html
% Table 8 — Period Probabilities of Death Within One Year (qx) at Selected Exact Ages, by Sex and Calendar Year (Cont.)
% The raw data from there is
%          Sex and Exact Age
%     |  Male                                                Female
% Year| [0 30 60 65 70 100]                                  [0 30 60 65 70 100]
% 2010| [0.00587,0.00116,0.01086,0.01753,0.02785,0.39134]    [0.00495,0.00060,0.00734,0.01201,0.01912,0.34031]
% I just take the numbers for Males, and then set my actual values based on a linear interpolation of the data.
dj_temp=interp1([0,30,60,65,70,100],[0.00587,0.00116,0.01086,0.01753,0.02785,0.39134],0:1:100,'linear');
Params.sj=1-dj_temp(20:100);
Params.sj(1)=1;
Params.sj(end)=0;
% I use linear interpolation to fill in the numbers inbetween those reported by Bell & Miller (2005).
% I have additionally imposed that the prob of death at age 20 be zero and that prob of death at age 100 is one.

%% Grids and exogenous shocks
% Grid for housing, from 0 to 20
minh=0; % min for housing; note that h'=0 means renter
maxh=20; % maximum value of housing: 
h_grid=minh + (maxh-minh)*linspace(0,1,n_a(2))'.^2; % note: ^2 is adding small amount of curvature, so most points are near hmin

% Note: -(1-Params.gamma)*maxh is the minimum possible value of assets, and maxassets is the maximum possible value
minassets=-(1-Params.gamma)*maxh;
maxassets=100; % maximum value of assets (Chen 2010 codes uses 100)
% First, the negative part of the grid: equally spaced from -(1-Params.gamma)*maxh up to 0. 
n_a_neg = round(0.1*n_a(1)); % use 1/10 of points in the negative assets range
asset_grid_neg = linspace(minassets,0,n_a_neg)';
% Second, the positive part of the grid (more points nearer 0)
asset_grid_pos=maxassets*linspace(0,1,n_a(1)-n_a_neg+1)'.^3;
asset_grid = [asset_grid_neg(1:n_a_neg-1); asset_grid_pos]; % Note: both contain zero, so omit it from asset_grid_neg
a_grid=[asset_grid; h_grid]; % stacked column vector

kfttoptions.initialj1sigmaz=Params.sigma_z1;
kfttoptions.nSigmas=2; % plus/minus 2 std deviations as the max/min grid points
[z_grid_J, pi_z_J,jequaloneDistz,otheroutputs] = discretizeLifeCycleAR1_KFTT(0,Params.rho_z,Params.sigma_z_epsilon,n_z,N_j,kfttoptions);
z_grid_J=exp(z_grid_J);

d_grid=[]; % no d variable

%% Return fn
DiscountFactorParamNames={'beta','sj'};

ReturnFn=@(aprime,hprime,a,h,z,kappaj,r,tau_p,theta,upsilon,sigma,gamma,phi,alpha,delta_k,delta_o,delta_r,agej,Jr,b)...
    LifeCycleModel40_ReturnFn(aprime,hprime,a,h,z,kappaj,r,tau_p,theta,upsilon,sigma,gamma,phi,alpha,delta_k,delta_o,delta_r,agej,Jr,b);

%% Solve the value function
% With two standard endogenous states, divide-and-conquer and grid interpolation
% layer are applied (only) to the first endogenous state. That's why we set
% up housing as the second endogenous state (see comment at top of file).
vfoptions.divideandconquer=1; % turn on divide-and-conquer
vfoptions.gridinterplayer=1; % turn on grid interpolation layer
vfoptions.ngridinterp=20; % 20 evenly-spaced points between each pair of consecutive grid points on the first endogenous state
simoptions.gridinterplayer=vfoptions.gridinterplayer; % grid interpolation layer must also be set in simoptions (because it changes Policy size/interpretation)
simoptions.ngridinterp=vfoptions.ngridinterp;

tic;
[V,Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
vftime=toc


%% Age distribution
AgeWeightParamNames={'mewj'};
Params.mewj=ones(1,N_j);
for jj=1:N_j-1
    Params.mewj(jj+1)=Params.mewj(jj)*Params.sj(jj)/(1+Params.n); % iteratively apply formula that mewj is determined by conditional survival probabilities (sj) and population growth rate (n)
end
Params.mewj=Params.mewj/sum(Params.mewj); % Normalize total mass to one

%% Initial age 20 (period 1) distribution
jequaloneDist=zeros([n_a,n_z],'gpuArray');
% Everyone is born with zero assets and zero housing
[~,zeroassetindex]=min(abs(asset_grid));
jequaloneDist(zeroassetindex,1,:)=shiftdim(jequaloneDistz,-2); % initial dist of z, with zero assets and zero housing

%% Agent distribution
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Params,simoptions);

%% AllStats
FnsToEvaluate.A=@(aprime,hprime,a,h,z) a;
FnsToEvaluate.N=@(aprime,hprime,a,h,z,kappaj) kappaj*z;
FnsToEvaluate.H=@(aprime,hprime,a,h,z) h; % housing
FnsToEvaluate.Hr=@(aprime,hprime,a,h,z,kappaj,r,tau_p,theta,upsilon,phi,alpha,delta_k,delta_o,delta_r,agej,Jr,b) (hprime==0)*LifeCycleModel40_HousingServicesFn(aprime,hprime,a,h,z,kappaj,r,tau_p,theta,upsilon,phi,alpha,delta_k,delta_o,delta_r,agej,Jr,b); % rental housing=housing services used by renters
FnsToEvaluate.pensiontaxrevenue=@(aprime,hprime,a,h,z,tau_p,w,kappaj) tau_p*w*kappaj*z;
FnsToEvaluate.pensionspend=@(aprime,hprime,a,h,z,agej,Jr,b) (agej>=Jr)*b;
FnsToEvaluate.accidentalbeqleft=@(aprime,hprime,a,h,z,r,sj,delta_o) (1+r)*aprime*(1-sj)+(1-delta_o)*hprime*(1-sj);

AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist,Policy, FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,simoptions);

% Note: Because labor supply is exogenous we can set b to clear the general eqm constraint directly, rather than solving for it as part of general eqm
Params.b=Params.b*(AllStats.pensiontaxrevenue.Mean/AllStats.pensionspend.Mean);

Params.p=(Params.r+Params.delta_r)/(1+Params.r); % price of housing
Params.w=(1-Params.alpha)*((Params.r+Params.delta_k)/Params.alpha)^(Params.alpha/(Params.alpha-1)); % wage

%% We will compute some stats conditional on being a homeowner
simoptions.conditionalrestrictions.Homeowners=@(aprime,hprime,a,h,z) (hprime>0);

% Some additional outputs
FnsToEvaluate.Homeownership=@(aprime,hprime,a,h,z) (hprime>0);
FnsToEvaluate.TotalWealth=@(aprime,hprime,a,h,z) a+h;
FnsToEvaluate.LoanToValueRatio=@(aprime,hprime,a,h,z) (aprime<0)*abs(aprime/hprime);
FnsToEvaluate.earnings=@(aprime,hprime,a,h,z,w,kappaj) w*kappaj*z;
FnsToEvaluate.housingservices=@(aprime,hprime,a,h,z,kappaj,r,tau_p,theta,upsilon,phi,alpha,delta_k,delta_o,delta_r,agej,Jr,b) LifeCycleModel40_HousingServicesFn(aprime,hprime,a,h,z,kappaj,r,tau_p,theta,upsilon,phi,alpha,delta_k,delta_o,delta_r,agej,Jr,b);

AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist,Policy, FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,simoptions);

K=AllStats.A.Mean-AllStats.Hr.Mean*(1-Params.p); % physical capital
Y=(K^Params.alpha)*(AllStats.N.Mean^(1-Params.alpha)); % output

fprintf('Quantitative properties of the benchmark economy \n')
fprintf('r is %1.2f%% \n',100*Params.r)
fprintf('K/Y is %1.3f \n',K/Y)
fprintf('H/Y is %1.3f \n',AllStats.H.Mean/Y)
fprintf('Homeownership rate is %2.1f%% \n', 100*AllStats.Homeownership.Mean)
fprintf('Gini for total wealth is %1.2f \n',AllStats.TotalWealth.Gini) 
fprintf('Gini for financial wealth is %1.2f \n',AllStats.A.Gini)
fprintf('Gini for housing is %1.2f \n',AllStats.H.Gini)
fprintf('Mean loan-to-value ratio (for borrowers) is %2.1f \n',100*AllStats.Homeowners.LoanToValueRatio.Mean)


% Compute some stats for working-age agents
simoptions.agegroupings=[1,Params.Jr]; % working age and retirees
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,simoptions);
AgeConditionalStats.earnings.Gini(1) % is 0.33, decent way below 0.4
AllStats.earnings.Gini % 0.45, so still not 0.4

% Plot some life-cycle profiles to see more about what is going on
simoptions=struct();
AgeConditionalStats2=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,simoptions);

% Lets also graph some life-cycle profiles based on 5-year age bins. 
simoptions.agegroupings=1:5:N_j; % 5-period bins [numbers are the start of each bin]
AgeConditionalStats5=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,simoptions);

figure(1)
subplot(3,1,1); plot(Params.agejshifter+Params.agej,AgeConditionalStats2.earnings.Mean)
title('Earnings (age conditional mean of)')
subplot(3,1,2); plot(Params.agejshifter+Params.agej,AgeConditionalStats2.A.Mean)
title('Financial Wealth (age conditional mean of)')
subplot(3,1,3); plot(Params.agejshifter+Params.agej,AgeConditionalStats2.H.Mean)
title('Housing (age conditional mean of)')

figure(2)
plot(Params.agejshifter+Params.agej,AgeConditionalStats2.consumption.Mean)
hold on
plot(Params.agejshifter+Params.agej,AgeConditionalStats2.housingservices.Mean)
plot(Params.agejshifter+Params.agej,AgeConditionalStats2.earnings.Mean)
hold off
legend('consumption','housing services', 'earnings')
title('Consumption, Housing Services and Earnings')

% Plot home-ownership based on 5-year age bins
figure(3)
plot(20:5:85,AgeConditionalStats5.Homeownership.Mean)
title('Home-ownership rate (5yr age bins)')


