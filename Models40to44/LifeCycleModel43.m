%% Life-Cycle Model 43: experienceassetz (Earnings-Indexed Pensions)
% 'experienceassetz' is when aprime(d,a,z) --- when next period endogenous
% state cannot be chosen directly, but is instead a function of a decision
% variable, this period endogenous state, and the exogenous markov state.

% Every model so far has given retirees a pension that is just a parameter,
% the same for everyone. Some pension systems pay you based on what you
% earned while working. So here the second endogenous state is the agents
% lifetime average earnings, and the pension is a function of it.

% Earnings are w*kappa_j*z*h, so they depend on the exogenous shock z. This
% is exactly why we need experienceassetz rather than the experienceasset
% of Life-Cycle Model 41: with aprime(d,a) we could not write down how the
% entitlement evolves, because the shock is part of it.

% This model has two endogenous states. VFI Toolkit requires that the
% experienceassetz is the later of the two.

% Because next period endogenous state for the experienceassetz is not
% chosen directly, it does not get included as an input to ReturnFn nor
% FnsToEvaluate.

%% How does VFI Toolkit think about this?
%
% One decision variable: h, fraction of time worked
% Two endogenous state variables: a, assets (total household savings), and ebar, lifetime average earnings
% One stochastic exogenous state variable: z, an AR(1) process (in logs), idiosyncratic shock to labor productivity units
% Age: j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=51; % Endogenous labour choice (fraction of time worked)
n_a=[201,31]; % Endogenous asset holdings, lifetime average earnings
n_z=21; % Exogenous labor productivity units shock
N_j=Params.J; % Number of periods in finite horizon

%% Parameters

% Discount rate
Params.beta = 0.96;
% Preferences
Params.sigma = 2; % Coeff of relative risk aversion (curvature of consumption)
Params.eta = 1.5; % Curvature of leisure (This will end up being 1/Frisch elasticity)
Params.psi = 10; % Weight on leisure

% Prices
Params.w=1; % Wage
Params.r=0.05; % Interest rate (0.05 is 5%)

% Demographics
Params.agej=1:1:Params.J; % Is a vector of all the agej: 1,2,3,...,J
Params.Jr=46;

% Pensions
% The pension is a piecewise-linear function of lifetime average earnings,
% with two 'bend points'. This is how the US social security system works:
% you get 90 cents of pension per dollar of average earnings up to the
% first bend point, 32 cents per dollar between the two bend points, and 15
% cents per dollar above the second. So the system is progressive.
Params.pensionbp1=0.2; % first bend point
Params.pensionbp2=1.2; % second bend point
Params.pensionrate1=0.9; % marginal replacement rate below the first bend point
Params.pensionrate2=0.32; % marginal replacement rate between the bend points
Params.pensionrate3=0.15; % marginal replacement rate above the second bend point
Params.pensionscale=1; % scales the whole pension formula
Params.pensionmin=0.05; % a minimum pension, paid even to someone who never worked

% Life-cycle profile of earnings
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];

% Exogenous shock process: AR1 on labor productivity units
Params.rho_z=0.9;
Params.sigma_epsilon_z=0.03;

% Conditional survival probabilities: sj is the probability of surviving to be age j+1, given alive at age j
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
Params.wg1=0.3; % (relative) importance of bequests
Params.wg2=3; % degree to which bequests are a luxury good (>=1; =1 would be a normal good)
Params.wg3=Params.sigma; % By using the same curvature as the utility of consumption it makes it much easier to guess appropriate parameter values for the warm glow

%% Grids
% The ^3 means that there are more points near 0. We know from theory that the value function will be more 'curved' near zero assets,
% and putting more points near curvature (where the derivative changes the most) increases accuracy of results.
asset_grid=10*(linspace(0,1,n_a(1)).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.

ebar_grid=linspace(0,2,n_a(2))'; % Because ebar is an experienceassetz, it will be interpolated onto this grid and so we need less grid points than usual
% Note: ebar_grid starts at zero, someone who never works has zero lifetime average earnings

% First, the AR(1) process z
[z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_epsilon_z,n_z);
z_grid=exp(z_grid); % Take exponential of the grid
[mean_z,~,~,~]=MarkovChainMoments(z_grid,pi_z); % Calculate the mean of the grid so as can normalise it
z_grid=z_grid./mean_z; % Normalise the grid on z (so that the mean of z is exactly 1)

% Grid for labour choice
h_grid=linspace(0,1,n_d)'; % Notice that it is imposing the 0<=h<=1 condition implicitly
d_grid=h_grid;
% Put asset and lifetime-average-earnings grids together
a_grid=[asset_grid; ebar_grid]; % 'stacked-column vector', so one column vector on top of the other

%% experienceassetz: aprimeFn
% To use an experienceassetz, we need to define aprime(d,a,z)
% [in notation of the current model, ebarprime(h,ebar,z)]

vfoptions.experienceassetz=1; % Using an experience-asset-z
% Note: by default, assumes it is the last d variable that controls the
% evolution of the experience asset (and that the last a variable is
% the experience asset).

% aprimeFn gives the value of ebarprime
% While working, ebar is the running average of earnings so far: at age agej
% the new average is the old one, plus 1/agej of the gap between this
% periods earnings and the old average. Once retired the entitlement is
% frozen (you stop accruing).
vfoptions.aprimeFn=@(h,ebar,z,w,kappa_j,agej,Jr) (agej<Jr)*(ebar+(w*kappa_j*z*h-ebar)/agej)+(agej>=Jr)*ebar;
% The first three inputs must be (d,a,z) [in the sense of aprime(d,a,z)], then any parameters

% We also need to tell simoptions about the experienceassetz
simoptions.experienceassetz=1;
simoptions.aprimeFn=vfoptions.aprimeFn;
simoptions.d_grid=d_grid; % Needed to handle aprimeFn
simoptions.a_grid=a_grid; % Needed to handle aprimeFn

% Use divide-and-conquer and grid interpolation layer on the standard endogenous state, assets (see Life-Cycle Models 29 and 30)
vfoptions.divideandconquer=1; % turn on divide-and-conquer
vfoptions.gridinterplayer=1; % turn on grid interpolation layer
vfoptions.ngridinterp=20; % 20 evenly-spaced points between each pair of consecutive a_grid points
% vfoptions.lowmemory=1; % default=0, set =1 to use loops (over z) if you get a gpu out-of-memory error, the loops reduce memory use but slow the runtimes
simoptions.gridinterplayer=vfoptions.gridinterplayer; % grid interpolation layer must also be set in simoptions (because it changes Policy size/interpretation)
simoptions.ngridinterp=vfoptions.ngridinterp;

% To see what the pension system looks like, here is the pension as a function of lifetime average earnings
pension_ebar=Params.pensionscale*(Params.pensionrate1*min(ebar_grid,Params.pensionbp1)+Params.pensionrate2*max(min(ebar_grid,Params.pensionbp2)-Params.pensionbp1,0)+Params.pensionrate3*max(ebar_grid-Params.pensionbp2,0));
pension_ebar=max(pension_ebar,Params.pensionmin);
figure(1)
plot(ebar_grid,pension_ebar,ebar_grid,ebar_grid)
legend('pension','45 degree','location','northwest')
title('Pension as a function of lifetime average earnings (ebar)')
xlabel('Lifetime average earnings (ebar)')
ylabel('Pension')
% Notice the two kinks, at the bend points, and that the pension is flat at
% pensionmin for the very lowest earners. Because the slope falls as you
% earn more, the system redistributes from high to low lifetime earners.

%% Now, create the return function
DiscountFactorParamNames={'beta','sj'};

% Use 'LifeCycleModel43_ReturnFn'
ReturnFn=@(h,aprime,a,ebar,z,w,sigma,psi,eta,agej,Jr,r,kappa_j,pensionbp1,pensionbp2,pensionrate1,pensionrate2,pensionrate3,pensionscale,pensionmin,wg1,wg2,wg3,beta,sj)...
    LifeCycleModel43_ReturnFn(h,aprime,a,ebar,z,w,sigma,psi,eta,agej,Jr,r,kappa_j,pensionbp1,pensionbp2,pensionrate1,pensionrate2,pensionrate3,pensionscale,pensionmin,wg1,wg2,wg3,beta,sj);
% Notice how we have (h,aprime,a,ebar,z,...)
% Follow same decision-next endo-endo-exo ordering as usual, but because ebar
% is an experienceassetz, we do not include ebarprime as it is not chosen
% directly.
% Notice also that 'pension' is no longer one of the parameters. The
% pension is now calculated inside the return function, out of ebar.

%% Solve the value function iteration problem
disp('Solve for Value fn and Policy fn using ValueFnIter command')
vfoptions.verbose=1; % give feedback
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

% V is now (a,ebar,z,j)
% Compare
size(V)
% with
[n_a,n_z,N_j]
% there are the same.
% Policy is
size(Policy)
% which is the same as
[length(n_d)+length(n_a)-1,n_a,n_z,N_j]
% minus one because the next period experienceassetz cannot be chosen directly

%% Let's take a quick look at what we have calculated, namely V and Policy

% Convert the policy function to values (rather than indexes).
% Policy(1,:,:,:,:) is h, Policy(2,:,:,:,:) is aprime [as function of (a,ebar,z,j)]
% Because ebar is an experienceassetz, ebarprime is not chosen directly so is not in Policy
% When using grid interpolation layer there is also a Policy(3,:,:,:,:) that is related to aprime on the interpolation layer (not something you need to understand as user, just mentioning)

% Plots are conditional on median z, and on two different levels of entitlement
zind=ceil(n_z/2);
ebarind_low=ceil(n_a(2)/6);
ebarind_high=ceil(5*n_a(2)/6);
PolicyVals=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions);
figure(2)
subplot(2,1,1); surf(asset_grid*ones(1,Params.J),ones(n_a(1),1)*(1:1:Params.J),reshape(PolicyVals(1,:,ebarind_low,zind,:),[n_a(1),Params.J]))
title('Policy function: fraction of time worked (h), low entitlement, median z')
xlabel('Assets (a)')
ylabel('Age j')
zlabel('Fraction of time worked (h)')
subplot(2,1,2); surf(asset_grid*ones(1,Params.J),ones(n_a(1),1)*(1:1:Params.J),reshape(PolicyVals(1,:,ebarind_high,zind,:),[n_a(1),Params.J]))
title('Policy function: fraction of time worked (h), high entitlement, median z')
xlabel('Assets (a)')
ylabel('Age j')
zlabel('Fraction of time worked (h)')
% Comparing the two is the whole point of the model. An agent with a low
% entitlement is below the first bend point, so an extra dollar of earnings
% today buys 90 cents of pension. An agent with a high entitlement is above
% the second bend point and only gets 15 cents. So the pension system gives
% the two of them quite different incentives to work.

% Because ebar is an experienceassetz, ebarprime is not chosen directly so is not in Policy
% But then how does ebarprime evolve? Remember that ebarprime(h,ebar,z), so you can
% use the current ebar, together with h (which is in Policy) and the current z,
% and you would then need to pass these as inputs into aprimeFn, which
% outputs the value of ebarprime(h,ebar,z). That is what the toolkit is doing
% internally (as well as then using linear interpolation to put the value
% of ebarprime(h,ebar,z) back onto the two nearest points on ebar_grid)

%% Now, we want to graph Life-Cycle Profiles

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are at age j=1.
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,1,floor((n_z+1)/2))=1; % All agents start with zero assets, zero pension entitlement, and the median shock

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

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (h,aprime,a,ebar,z) as first inputs, then just any relevant parameters.
FnsToEvaluate.fractiontimeworked=@(h,aprime,a,ebar,z) h; % h is fraction of time worked
FnsToEvaluate.earnings=@(h,aprime,a,ebar,z,w,kappa_j) w*kappa_j*z*h; % w*kappa_j*z*h is the labor earnings
FnsToEvaluate.entitlement=@(h,aprime,a,ebar,z) ebar; % ebar is lifetime average earnings so far
FnsToEvaluate.assets=@(h,aprime,a,ebar,z) a; % a is the current asset holdings
FnsToEvaluate.pension=@(h,aprime,a,ebar,z,agej,Jr,pensionbp1,pensionbp2,pensionrate1,pensionrate2,pensionrate3,pensionscale,pensionmin) (agej>=Jr)*max(pensionscale*(pensionrate1*min(ebar,pensionbp1)+pensionrate2*max(min(ebar,pensionbp2)-pensionbp1,0)+pensionrate3*max(ebar-pensionbp2,0)),pensionmin); % the pension actually received (zero while working)
% notice that we have called these fractiontimeworked, earnings, entitlement, assets and pension

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

%% Plot the life cycle profiles
figure(3)
subplot(5,1,1); plot(Params.agejshifter+(1:1:Params.J),AgeConditionalStats.fractiontimeworked.Mean)
title('Life Cycle Profile: Fraction Time Worked (h)')
subplot(5,1,2); plot(Params.agejshifter+(1:1:Params.J),AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Labor Earnings (w kappa_j z h)')
subplot(5,1,3); plot(Params.agejshifter+(1:1:Params.J),AgeConditionalStats.entitlement.Mean)
title('Life Cycle Profile: Lifetime Average Earnings (ebar)')
subplot(5,1,4); plot(Params.agejshifter+(1:1:Params.J),AgeConditionalStats.pension.Mean)
title('Life Cycle Profile: Pension')
subplot(5,1,5); plot(Params.agejshifter+(1:1:Params.J),AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Assets (a)')
% The entitlement rises through working life and is then flat in
% retirement, which is just what the law of motion says it should do.

%% Because the pension now depends on your own earnings history, retirees are no longer identical in their pension income
fprintf('At the first year of retirement: pension mean %1.4f, Gini %1.4f \n',AgeConditionalStats.pension.Mean(Params.Jr),AgeConditionalStats.pension.Gini(Params.Jr))
fprintf('At the first year of retirement: lifetime average earnings mean %1.4f, Gini %1.4f \n',AgeConditionalStats.entitlement.Mean(Params.Jr),AgeConditionalStats.entitlement.Gini(Params.Jr))
% The pension Gini should be smaller than the lifetime-average-earnings
% Gini. That is the progressivity of the bend-point formula showing up: the
% pension system compresses the earnings distribution.
