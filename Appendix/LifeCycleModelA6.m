%% Life-Cycle Model A6: Two i.i.d. (e) shocks, independent
% This will be nice and simple, as using two i.i.d. shocks is hopefully 
% 'obvious' now that we have covered two markov shocks in models A5i-A5iv.
%
% Use two exogenous shocks: e1 and e2, both are shocks to labor efficiency units
% We have to change ReturnFn and FnsToEvaluate inputs to take (...,e1,e2,...)
% We start with a simple example where we just have two independent i.i.d.
% e1 is normal distributions, and as in Life-Cycle Model 9, 
% we just use Tanaka-Toda method to discretize them which gives us e1_grid, 
% and pi_e1 (note we just use the first row of the transition
% matrices as pi_e1, since we are dealing with i.i.d variables). e2 is
% uniform, so we can easily manually discretize this.
% 
% We put the grids together as a stacked column vector
% e_grid=[e1_grid; e2_grid]. 
% Note that we could alternatively use a joint-grid.
%
% We put the transition matrices together as 
% pi_e=kron(pi_e2,pi_e1);
% Notice that this is the kron(), with pi_e1 and pi_e2 in reverse order.
%
% So we have size(pi_e)=sum(n_e)-by-1 and size(pi_e)=prod(n_e)-by-1
%
% How to interpret pi_e?
% Say e1 has three values [e1a;e1b;e1c], and e2 has two values [e2a;e2b]
% Consider putting them together, so that we count through all the e1
% values, keeping e2 value fixed, and once we complete all e1 values we
% increment e2 value and start counting through e1 values again.
% So think of
% [e1a, e2a;
%  e1b, e2a;
%  e1c, e2a;
%  e1a, e2b;
%  e1b, e2b;
%  e1c, e2b]
% This is what the dimensions of pi_e are capturing. The rows are this
% period and the columns are next period (as always for our markov
% transition matrices).
% So for example the 2nd row, 3rd column element in pi_e represents the
% probability of going from (e1b,e2a) today, to (e1c, e2a) tomorrow.
% Another example, the 5th row, 2nd column element in pi_e represents the
% probability of going from (e1b,e2b) today, to (e1b, e2a) tomorrow.
%
% These three things:
% i) change inputs of ReturnFn and FnsToEvaluate to include (...,e1,e2,...)
% ii) create e_grid (here as a stacked column as they are independent)
% iii) create pi_e (here as kron() of reverse order, as they are independent)
% are the only changes we need to make to the code.
% Note, we still put pi_e and e_grid into both vfoptions and simoptions.
%
% If we want to use age-dependent i.i.d shocks, or we want to use
% joint-grids instead of stacked column vectors, or we want to have the two
% shocks probabilities depend on each other, all of this can be done
% analagous to how we did it for markov shocks.
% Note: you can find an age-dependent i.i.d shock in Life-Cycle model 27,
% and you can find correlated i.i.d. shocks in Life-Cycle model 28.
%
% Note: The above explains e_grid and pi_e with 3 points on e1 and 2 points
% on e2. The code below uses 5 points on e1 and 3 points on e2. Take a look
% at e_grid and pi_e, see if they make sense to you (what size they are,
% and how to interpret them).
%
% Unimportant comment: Notice that the viewed from 'inside' the ReturnFn
% there is no distinction between different kinds of shocks. Hence we can
% just reuse LifeCycleModelA5_ReturnFn(), and where we previously put two
% markovs we now just put two i.i.d.


%% How does VFI Toolkit think about this?
%
% One decision variable: h, labour hours worked
% One endogenous state variable: a, assets (total household savings)
% Two stochastic exogenous state variables: z1 and z2, both are shocks to labor efficiency units
% Age: j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=51; % Endogenous labour choice (fraction of time worked)
n_a=201; % Endogenous asset holdings
n_z=0;
n_e=[5,3]; % Exogenous labor productivity units shock, two of them
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
% Age-dependent labor productivity units
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
% Exogenous shock process, e1: i.i.d. normal on labor productivity units
Params.mew_e1=0;
Params.sigma_epsilon_e1=0.03;
% Exogenous shock process, e2: i.i.d. uniform on labor productivity units
Params.e2_min=0.5;
Params.e2_max=1.5;

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

% Grid for labour choice
h_grid=linspace(0,1,n_d)'; % Notice that it is imposing the 0<=h<=1 condition implicitly
% Switch into toolkit notation
d_grid=h_grid;

% n_z=0, create the following as they are still required as inputs to some commands
z_grid=[];
pi_z=[];

%% Dealing with two i.i.d.: grid and transition matrix, e_grid and pi_e
% First, just discretize each of our two i.i.d. normal processes
% Discretize the i.i.d. normal e1
[e1_grid,pi_e1]=discretizeAR1_FarmerToda(Params.mew_e1,0,Params.sigma_epsilon_e1,n_e(1));
e1_grid=exp(e1_grid); % Take exponential of the grid
[mean_e1,~,~,~]=MarkovChainMoments(e1_grid,pi_e1); % Calculate the mean of the grid so as can normalise it
e1_grid=e1_grid./mean_e1; % Normalise the grid on e1 (so that the mean of e1 is exactly 1)
pi_e1=pi_e1(1,:)'; % i.i.d., so just use first row of matrix (and transpose it to column)

% Discretize the i.i.d. uniform e2
e2_grid=linspace(Params.e2_min,Params.e2_max,n_e(2))';
pi_e2=ones(n_e(2),1)/n_e(2); % equal probability on each point

% Now, we put together the two grids, as a stacked column
e_grid=[e1_grid; e2_grid];

% Next, we put together pi_e
pi_e=kron(pi_e2,pi_e1); % note reverse order

% For i.i.d shocks, we have to tell the toolkit about them using vfoptions and simoptions
vfoptions.n_e=n_e;
vfoptions.e_grid=e_grid;
vfoptions.pi_e=pi_e;
simoptions.n_e=vfoptions.n_e;
simoptions.e_grid=vfoptions.e_grid;
simoptions.pi_e=vfoptions.pi_e;

%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

% Use 'LifeCycleModelA5_ReturnFn', which has two i.i.d. exogenous states
ReturnFn=@(h,aprime,a,e1,e2,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)... 
    LifeCycleModelA5_ReturnFn(h,aprime,a,e1,e2,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)
% Note: From the perspective of the ReturnFn, it does not know what kind of
% states e1 and e2 are (whether they are markov, i.i.d. or anything else).
% Hence we can just reuse LifeCycleModelA5_ReturnFn which was originally
% for two markov shocks, but here it will be two i.i.d shocks.

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
% vfoptions=struct(); % Just using the defaults.
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

% V is now (a,e1,e2,j). One dimension for each state variable.
% Compare
size(V)
% with
[n_a,n_e(1),n_e(2),N_j]
% there are the same.
% Policy is
size(Policy)
% which is the same as
[length(n_d)+length(n_a),n_a,n_e(1),n_e(2),N_j]
% The n_a,n_e(1),n_e(2),N_j represent the state on which the decisions/policys
% depend, and there is one decision for each decision variable 'd' and each
% endogenous state variable 'a', and one for each exogenous i.i.d state variable 'e'

%% We won't plot the Value and Policy fn, but thinking out how you would might be a good way to check you understand the form of V and Policy

%% Now, we want to graph Life-Cycle Profiles

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are
% at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_e],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,floor((n_e(1)+1)/2),floor((n_e(2)+1)/2))=1; % All agents start with zero assets, and the median shock

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
% simoptions=struct(); % Use the default options
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);


%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (h,aprime,a,z) as first
% inputs, then just any relevant parameters.
FnsToEvaluate.fractiontimeworked=@(h,aprime,a,e1,e2) h; % h is fraction of time worked
FnsToEvaluate.earnings=@(h,aprime,a,e1,e2,w,kappa_j) w*kappa_j*h*e1*e2; % w*kappa_j*h*e1*e2 is the labor earnings (note: h will be zero when z is zero, so could just use w*kappa_j*h)
FnsToEvaluate.assets=@(h,aprime,a,e1,e2) a; % a is the current asset holdings

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
title('Life Cycle Profile: Fraction Time Worked (h)')
subplot(3,1,2); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Labor Earnings (w kappa_j h)')
subplot(3,1,3); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Assets (a)')



