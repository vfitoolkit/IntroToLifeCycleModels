%% Life-Cycle Model 12: Epstein-Zin preferences
% Epstein-Zin preferences use seperate parameters to determine risk aversion and the intertemporal elasticity of substitution.

%% Use Epstein-Zin preferences. I have put all the lines that relate to Epstein-Zin preferences together to simplify reading.
% There are essentially three parts to using Epstein-Zin preferences.
% 1. Use vfoptions to state that you are using Epstein-Zin preferences.
% 2. Set the appropriate preference parameters.
% 3. Minor adjustment to 'discount factors' and 'return function'.
% 4. Warm glow of bequests

% 1. Use vfoptions to state that you are using Epstein-Zin preferences.
vfoptions.exoticpreferences='EpsteinZin'; %Use Epstein-Zin preferences
% EZ preferences are different for positive and negative valued utility
% functions, by default it would be assumed that the utility is negative,
% but let's just be explicit and specify this anyway
vfoptions.EZpositiveutility=0; % If utility function was positive valued, you would need to set this to one

% 2. Set the appropriate preference parameters.
% Epstein-Zin preference parameters
vfoptions.EZriskaversion='phi'; % Name of the relative risk aversion parameter
Params.phi=-1;

% 3. Minor adjustment to 'discount factors' and 'return function'.
% To be able to use a warm-glow-of-bequests with Epstein-Zin preferences we have to distingush
% conditional survival probabilities from the regular discount factor. 
% So below the discount factor is now just beta, and we add
% vfoptions.survivalprobability='sj'
% We no longer include warm-glow of bequests in the return fn.

% 4. Warm-glow of bequests
% Using warm-glow of bequests together with EZ preferences is subtle, so dealing with them has been mostly automated.
% Need to define two things (if you don't want bequests you simply do not define these)
vfoptions.WarmGlowBequestsFn=@(aprime,sigma,wg,agej,Jr) (agej>=Jr+10)*wg*(aprime^(1-sigma))/(1-sigma); % First input arguement must be aprime, after than can be any parameters
% Comment: Loosely speaking you want the WarmGlowBequestsFn to output the 'same'
% thing as the return fn. Our utility function has (c^(1-sigma))/(1-sigma)
% and hence we set the warmglow to (aprime^(1-sigma))/(1-sigma). We can
% then control the importance of the warm-glow of bequests by multiplying
% it by a constant, here called wg. Note that to keep this in line with previous models we
% also include a term so that the warm-glow of bequests in only non-zero
% once we are 10 periods into the retirement, hence the (agej>=Jr+10)
% We declare the value of 'wg' below (there is also a comment on how to interpret wg).
% Comment: If a parameter in the WarmGlowBequestsFn depends on age, then it is the last period of life
% from which the parameter value is taken. So be careful. This may mean you want to create
% an offset version of a parameter to put into the WarmGlowBequestsFn. (E.g., if
% you die at the end of period 20, then it is the period 20 parameter
% values that will be used to evaluate WarmGlowBequestsFn; people often 
% think of the warm-glow as being received in the following period.)

% When using Epstein-Zin preferences the risk aversion is done as part of
% the value function iteration but not as part of the return function
% itself. This is in contrast to standard preferences when the risk
% aversion can just be done as part of the return function.

% That is all. Every other line of code is essentially unchanged!! Epstein-Zin really is that easy ;)

% See pdf for an explanation of how exactly warm-glow of bequests is being
% handled, as well as why it goes wrong if you just put warm-glow of
% bequests in the return function like you would with standard preferences.

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
Params.wg=10; % (relative) importance of bequests
% Note: wg can be interpreted as the target for the ratio of terminal wealth to terminal 
% consumption (terminal meaning at end of the last period, period J). See pdf for explanation.

%% Grids
% The ^3 means that there are more points near 0 and near 10. We know from
% theory that the value function will be more 'curved' near zero assets,
% and putting more points near curvature (where the derivative changes the most) increases accuracy of results.
a_grid=10*(linspace(0,1,n_a).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.

% First, the AR(1) process z
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
DiscountFactorParamNames={'beta'}; 
vfoptions.survivalprobability='sj';
% Note that because we have a warm-glow-of-bequests together with Epstein-Zin preferences we can no longer treat sj 
% as just another discount factor (if you have EZ preferences but no warm-glow you can just put sj as another discount factor)

% 'LifeCycleModel12_ReturnFn' is the same as was used for model 9, except we have to remove the warm-glow of bequests from the return
% fn as it has to be treated specially when using Epstein-Zin preferences.
% Note that the vfoptions.EZriskaversion parameter modifies this and makes it Epstein-Zin preferences.
ReturnFn=@(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j) ...
    LifeCycleModel12_ReturnFn(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j)

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
% vfoptions=struct(); % Just using the defaults.
vfoptions.lowmemory=1
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

% V is now (a,z,j). This was already true, just that previously z was trivial (a single point) 
% Compare
size(V)
% with
[n_a,n_z,N_j]
% there are the same.
% Policy is
size(Policy)
% which is the same as
[length(n_d)+length(n_a),n_a,n_z,N_j]
% The n_a,n_z,N_j represent the state on which the decisions/policys
% depend, and there is one decision for each decision variable 'd' and each endogenous state variable 'a'

% Check for hitting top of asset grid
temp=reshape(max(max(Policy(2,:,:,:),[],3),[],2),[1,N_j])

temp=reshape(max(max(Policy(1,:,:,:),[],3),[],2),[1,N_j])


%% Let's take a quick look at what we have calculated, namely V and Policy

% The value function V depends on the state, so now it depends on both asset holdings and age.

% We can plot V as a 3d plot (surf is matlab command for 3d plot)
% Which z value should we plot? I will plot the median
zind=floor(n_z+1)/2; % This will be the median
figure(1)
subplot(2,1,1); surf(a_grid*ones(1,Params.J),ones(n_a,1)*(1:1:Params.J),reshape(V(:,zind,:),[n_a,Params.J]))
title('Value function: median value of z')
xlabel('Assets (a)')
ylabel('Age j')
subplot(2,1,2); surf(a_grid*ones(1,Params.J),ones(n_a,1)*(Params.agejshifter+(1:1:Params.J)),reshape(V(:,zind,:),[n_a,Params.J]))
title('Value function: median value of z')
xlabel('Assets (a)')
ylabel('Age in Years')

% Do another plot of V, this time as a function (of assets) for a given age (I do a few for different ages)
figure(2)
subplot(5,1,1); plot(a_grid,V(:,1,1),a_grid,V(:,zind,1),a_grid,V(:,end,1)) % j=1
title('Value fn at age j=1')
legend('min z','median z','max z') % Just include the legend once in the top subplot
subplot(5,1,2); plot(a_grid,V(:,1,20),a_grid,V(:,zind,20),a_grid,V(:,end,20)) % j=20
title('Value fn at age j=20')
subplot(5,1,3); plot(a_grid,V(:,1,79),a_grid,V(:,zind,79),a_grid,V(:,end,79)) % j=79 % 45
title('Value fn at age j=79')
subplot(5,1,4); plot(a_grid,V(:,1,80),a_grid,V(:,end,80),a_grid,V(:,end,80)) % j=80 % 46
title('Value fn at age j=80 (first year of retirement)')
subplot(5,1,5); plot(a_grid,V(:,1,81),a_grid,V(:,zind,81),a_grid,V(:,end,81)) % j=81
title('Value fn at age j=81')
xlabel('Assets (a)')


figure(2)
subplot(5,2,1); plot(a_grid,V(:,1,60),a_grid,V(:,zind,60),a_grid,V(:,end,60)) % j=1
title('Value fn at age j=60')
legend('min z','median z','max z') % Just include the legend once in the top subplot
subplot(5,2,2); plot(a_grid,V(:,1,65),a_grid,V(:,zind,65),a_grid,V(:,end,65)) % j=20
title('Value fn at age j=65')
subplot(5,2,3); plot(a_grid,V(:,1,70),a_grid,V(:,zind,70),a_grid,V(:,end,70)) % j=79 % 45
title('Value fn at age j=70')
subplot(5,2,4); plot(a_grid,V(:,1,72),a_grid,V(:,end,72),a_grid,V(:,end,72)) % j=80 % 46
title('Value fn at age j=72')
subplot(5,2,5); plot(a_grid,V(:,1,74),a_grid,V(:,zind,74),a_grid,V(:,end,74)) % j=81
title('Value fn at age j=74')
subplot(5,2,6); plot(a_grid,V(:,1,76),a_grid,V(:,zind,76),a_grid,V(:,end,76)) % j=1
title('Value fn at age j=76')
subplot(5,2,7); plot(a_grid,V(:,1,78),a_grid,V(:,zind,78),a_grid,V(:,end,78)) % j=20
title('Value fn at age j=78')
subplot(5,2,8); plot(a_grid,V(:,1,79),a_grid,V(:,zind,79),a_grid,V(:,end,79)) % j=79 % 45
title('Value fn at age j=79')
subplot(5,2,9); plot(a_grid,V(:,1,80),a_grid,V(:,end,80),a_grid,V(:,end,80)) % j=80 % 46
title('Value fn at age j=80')
subplot(5,2,10); plot(a_grid,V(:,1,81),a_grid,V(:,zind,81),a_grid,V(:,end,81)) % j=81
title('Value fn at age j=81')
xlabel('Assets (a)')

% Convert the policy function to values (rather than indexes).
% Note that there is one policy for hours worked (h), and another for next period assets (aprime). 
% Policy(1,:,:,:) is h, Policy(2,:,:,:) is aprime [as function of (a,z,j)]
% Plot both as a 3d plot, again I arbitrarily choose the median value of z
figure(3)
PolicyVals=PolicyInd2Val_Case1_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions);
subplot(2,1,1); surf(a_grid*ones(1,Params.J),ones(n_a,1)*(1:1:Params.J),reshape(PolicyVals(1,:,zind,:),[n_a,Params.J]))
title('Policy function: fraction of time worked (h), median z')
xlabel('Assets (a)')
ylabel('Age j')
zlabel('Fraction of Time Worked (h)')
subplot(2,1,2); surf(a_grid*ones(1,Params.J),ones(n_a,1)*(1:1:Params.J),reshape(PolicyVals(2,:,zind,:),[n_a,Params.J]))
title('Policy function: next period assets (aprime), median z')
xlabel('Assets (a)')
ylabel('Age j')
zlabel('Next period assets (aprime)')

% Again, plot both policies (h and aprime), this time as a function (of assets) for a given age  (I do a few for different ages)
figure(4)
subplot(5,2,1); plot(a_grid,PolicyVals(1,:,1,1),a_grid,PolicyVals(1,:,zind,1),a_grid,PolicyVals(1,:,end,1)) % j=1
title('Policy for h at age j=1')
subplot(5,2,3); plot(a_grid,PolicyVals(1,:,1,20),a_grid,PolicyVals(1,:,zind,20),a_grid,PolicyVals(1,:,end,20)) % j=20
title('Policy for h at age j=20')
subplot(5,2,5); plot(a_grid,PolicyVals(1,:,1,45),a_grid,PolicyVals(1,:,zind,45),a_grid,PolicyVals(1,:,end,45)) % j=45
title('Policy for h at age j=45')
subplot(5,2,7); plot(a_grid,PolicyVals(1,:,1,46),a_grid,PolicyVals(1,:,zind,46),a_grid,PolicyVals(1,:,end,46)) % j=46
title('Policy for h at age j=46 (first year of retirement)')
subplot(5,2,9); plot(a_grid,PolicyVals(1,:,1,81),a_grid,PolicyVals(1,:,zind,81),a_grid,PolicyVals(1,:,end,81)) % j=81
title('Policy for h at age j=81')
xlabel('Assets (a)')
subplot(5,2,2); plot(a_grid,PolicyVals(2,:,1,1),a_grid,PolicyVals(2,:,zind,1),a_grid,PolicyVals(2,:,end,1)) % j=1
title('Policy for aprime at age j=1')
legend('min z','median z','max z') % Just include the legend once in the top-right subplot
subplot(5,2,4); plot(a_grid,PolicyVals(2,:,1,20),a_grid,PolicyVals(2,:,zind,20),a_grid,PolicyVals(2,:,end,20)) % j=20
title('Policy for aprime at age j=20')
subplot(5,2,6); plot(a_grid,PolicyVals(2,:,1,45),a_grid,PolicyVals(2,:,zind,45),a_grid,PolicyVals(2,:,end,45)) % j=45
title('Policy for aprime at age j=45')
subplot(5,2,8); plot(a_grid,PolicyVals(2,:,1,46),a_grid,PolicyVals(2,:,zind,46),a_grid,PolicyVals(2,:,end,46)) % j=46
title('Policy for aprime at age j=46 (first year of retirement)')
subplot(5,2,10); plot(a_grid,PolicyVals(2,:,1,81),a_grid,PolicyVals(2,:,zind,81),a_grid,PolicyVals(2,:,end,81)) % j=81
title('Policy for aprime at age j=81')
xlabel('Assets (a)')

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
simoptions=struct(); % Use the defaults (Epstein-Zin preferences affect the Policy, but once we have that they are irrelevant, except for welfare calculations)
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
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% For example
% AgeConditionalStats.earnings.Mean
% There are things other than Mean, but in our current deterministic model
% in which all agents are born identical the rest are meaningless.

%% Plot the life cycle profiles of fraction-of-time-worked, earnings, and assets

figure(5)
subplot(3,1,1); plot(1:1:Params.J,AgeConditionalStats.fractiontimeworked.Mean)
title('Life Cycle Profile: Fraction Time Worked (h)')
subplot(3,1,2); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Labor Earnings (w kappa_j h)')
subplot(3,1,3); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Assets (a)')



