%% Life-Cycle Model 28: Two decision variables (dual-earner household)
% Model of a household with two people. They share savings but can choose
% how much each of the two works. Each person has a persistent AR(1) and a
% transitory iid component to their labor efficiency units. The shocks to
% each person are correlated.
%
% Model shows how to do two decision variables.
% Model also usees 'joint grids' for the exogenous variables (see Life-Cycle Model A9).
%
% Use Farmer-Toda which produces a 'joint grid', as opposed to a cross-product (kronecker) grid.
% Extends Life-Cycle Model 11, adding the second earner to the household.
%
% Notice that while there are, e.g., two deterministic labor productivity
% earnings as a function of age (one for each earner), we do not use
% permament types as both are relevant to the one household.
%
% To keep the emphasis on the two decision variables (and the shock
% processes) we will keep everything else as little changed as possible, so
% we just add an extra term to the utility function for h2 that is
% identical to that for h1 (using the same parameters).
%
% Note: Typically in practice for this kind of model people want to capture
% that one spouse might choose not to work. This will almost never happen in the
% current setup but is easy to get just adding a fixed cost of working to
% the spouse (include a -fc*(h2>0) term in the budget constraint but
% modifying the return function, where fc is the fixed cost; you could model it 
% so the fixed cost is only incurred if both work).
%
% This example is a larger model than most, so to still be able to solve it
% on a gpu with 8gb gpu-memory we use vfoptions.lowmemory=1 (and
% vfoptions.paroverz=1) so that the value function is solved looping over
% the e variables but parallel over the z variables.

%% How does VFI Toolkit think about this?
%
% Two decision variable: h1 and h2, labour hours worked by each spouse
% One endogenous state variable: a, assets (total household savings)
% Four stochastic exogenous state variables: 
%     z1 and z2, both are AR(1) shocks to labor efficiency units
%     e1 and e2, both are iid shocks to labor efficiency units
% Age: j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=[21,21]; % Endogenous labour choice (fraction of time worked)
n_a=201; % Endogenous asset holdings
n_z=[3,3]; % Exogenous labor productivity units shock, two of them
n_e=[3,3];
vfoptions.n_e=n_e;
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
Params.kappa_j_1=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
Params.kappa_j_2=0.9*[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)]; % Note: person 2 faces lower earnings than person 1
% AR(1) processes on idiosyncratic labor productivity units with correlated innovations
Params.rho_z=[0.9,0;0,0.7];
Params.sigmasq_epsilon_z=[0.0303, 0.0027; 0.0027, 0.0382]; 
Params.sigma_epsilon_z=sqrt(Params.sigmasq_epsilon_z);
    % The Farmer-Toda method can discretize a VAR(1) with any (postivite semi-definite) variance-covariance matrix.
% iid processes on idiosyncratic labor units which are correlated
Params.sigmasq_epsilon_e=[0.1,0.05;0.05,0.1];
Params.sigma_epsilon_e=sqrt(Params.sigmasq_epsilon_e);

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

% First, the AR(1) process on z1 and z2 with correlated innovations.
% Notice that this is just a VAR(1) process on z1 and z2 (with zeros on the off-diagonals of the auto-correlation matrix)
% We use the Farmer-Toda method to discretize the VAR(1)
% Note that for VAR(1), the Farmer-Toda method produces a 'joint grid'
[z_grid, pi_z]=discretizeVAR1_FarmerToda([0;0],Params.rho_z,Params.sigma_epsilon_z,n_z);

z_grid=exp(z_grid);
% I skip normalizing this to 1 in the current model (would need to do each of z1 and z2 seperately)

% Second, the iid process on e1 and e2 which are correlated
% Notice that this is just a VAR(1) with zero auto-correlation
[e_grid, pi_e]=discretizeVAR1_FarmerToda(zeros(2,1),zeros(2,2),Params.sigma_epsilon_e,vfoptions.n_e);
e_grid=exp(e_grid);
pi_e=pi_e(1,:)';  % Because it is iid, the distribution is just the first row (all rows are identical). We use pi_e as a column vector for VFI Toolkit to handle iid variables.

% To use e variables we need to put them in vfoptions and simoptions
vfoptions.e_grid=e_grid;
vfoptions.pi_e=pi_e;
simoptions.n_e=vfoptions.n_e;
simoptions.e_grid=vfoptions.e_grid;
simoptions.pi_e=vfoptions.pi_e;

% Grid for labour choice
h1_grid=linspace(0,1,n_d(1))'; % Notice that it is imposing the 0<=h1<=1 condition implicitly
h2_grid=linspace(0,1,n_d(2))'; % Notice that it is imposing the 0<=h2<=1 condition implicitly
% Switch into toolkit notation
d_grid=[h1_grid; h2_grid];

%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

% Notice change to 'LifeCycleModel28_ReturnFn', and now input h1,h2 and z1,z2,e1,e2
ReturnFn=@(h1,h2,aprime,a,z1,z2,e1,e2,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j_1,kappa_j_2,warmglow1,warmglow2,warmglow3,beta,sj) LifeCycleModel28_ReturnFn(h1,h2,aprime,a,z1,z2,e1,e2,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j_1,kappa_j_2,warmglow1,warmglow2,warmglow3,beta,sj)

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
% Note: on a more powerful GPU you can set lowmemory=0 (which is the default) and things will run faster.
vfoptions.lowmemory=1;
vfoptions.paroverz=1;
vfoptions.verbose=1
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

% V is now (a,z1,z2,j). One dimension for each state variable.
% Compare
size(V)
% with
[n_a,n_z(1),n_z(2),N_j]
% there are the same.
% Policy is
size(Policy)
% which is the same as
[length(n_d)+length(n_a),n_a,n_z(1),n_z(2),N_j]
% The n_a,n_z(1),n_z(2),N_j represent the state on which the decisions/policys
% depend, and there is one decision for each decision variable 'd' and each
% endogenous state variable 'a', and one for each exogenous state variable 'z'

%% We won't plot the Value and Policy fn, but thinking out how you would might be a good way to check you understand the form of V and Policy

%% Now, we want to graph Life-Cycle Profiles

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are
% at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_z,n_e],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,floor((n_z(1)+1)/2),floor((n_z(2)+1)/2),floor((n_e(1)+1)/2),floor((n_e(2)+1)/2))=1; % All agents start with zero assets, and the median shock

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
% Again, we will explain in a later model what the stationary distribution
% is, it is not important for our current goal of graphing the life-cycle profile

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (h,aprime,a,z) as first inputs, then just any relevant parameters.
FnsToEvaluate.fractiontimeworked=@(h1,h2,aprime,a,z1,z2,e1,e2) h1+h2; % h is fraction of time worked
FnsToEvaluate.fractiontimeworked1=@(h1,h2,aprime,a,z1,z2,e1,e2) h1; % h is fraction of time worked
FnsToEvaluate.fractiontimeworked2=@(h1,h2,aprime,a,z1,z2,e1,e2) h2; % h is fraction of time worked
FnsToEvaluate.earnings=@(h1,h2,aprime,a,z1,z2,e1,e2,w,kappa_j_1,kappa_j_2) w*kappa_j_1*h1*z1*e1+w*kappa_j_2*h2*z2*e2; % w*kappa_j*h*z is the labor earnings (note: h will be zero when z is zero, so could just use w*kappa_j*h)
FnsToEvaluate.earnings1=@(h1,h2,aprime,a,z1,z2,e1,e2,w,kappa_j_1) w*kappa_j_1*h1*z1*e1; % w*kappa_j*h*z is the labor earnings (note: h will be zero when z is zero, so could just use w*kappa_j*h)
FnsToEvaluate.earnings2=@(h1,h2,aprime,a,z1,z2,e1,e2,w,kappa_j_2) w*kappa_j_2*h2*z2*e2; % w*kappa_j*h*z is the labor earnings (note: h will be zero when z is zero, so could just use w*kappa_j*h)
FnsToEvaluate.assets=@(h1,h2,aprime,a,z1,z2,e1,e2) a; % a is the current asset holdings

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
title('Life Cycle Profile: Fraction Time Worked (h1+h2)')
subplot(3,1,2); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Labor Earnings (w kappa_j_1 h1 z1 e1 + w kappa_j_2 h2 z2 e2)')
subplot(3,1,3); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Assets (a)')

figure(2)
subplot(2,1,1); plot(1:1:Params.J,AgeConditionalStats.fractiontimeworked.Mean,1:1:Params.J,AgeConditionalStats.fractiontimeworked1.Mean,1:1:Params.J,AgeConditionalStats.fractiontimeworked2.Mean)
title('Life Cycle Profile: Fraction Time Worked')
legend('h1+h2','h1','h2')
subplot(2,1,2); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean,1:1:Params.J,AgeConditionalStats.earnings1.Mean,1:1:Params.J,AgeConditionalStats.earnings2.Mean)
title('Life Cycle Profile: Labor Earnings')
legend('household','spouse 1','spouse 2')

