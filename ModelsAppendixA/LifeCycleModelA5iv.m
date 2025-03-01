%% Life-Cycle Model A5iv: Two markov (z) shocks, probabilities that depend on each other
% We will have two exogenous shocks, z1 and z2, each of which takes two possible values
% z2 is our 'macroeconomic' shock, it takes a value of 1 for 'recession' and 0 for 'expansion'
% z1 is our 'microeconomic' shock, it takes a value of 1 for 'employment' and 0 for 'unemployment'
%
% So we will have two markov z shocks, with independent grids, and we can
% just put these together as a stacked column vector.
% 
% Then we have a transition matrix pi_z2 for the macroeconomic shocks.
% And we want the transition probabilities for z1 to depend on the current
% z2 value.
% We create pi_z as being size(pi_z)=prod(n_z)-by-prod(n_z) as always.
% And we just have to think carefully about how set the probabilities in
% pi_z, see below for this where pi_z is created.



% The rest of this commentary is about the economics of this model:
%
% The idea is that agents care about the microeconomics shocks (which directly effect them), but 
% they do not directly care about the macroeconomic shocks (that only effect them indirectly). The
% macroeconomic shocks influence the microeconomic shocks (a recession makes unemployment more likely)
% which we model as the transition probabilities of the microeconomic shocks depending on the state of the macroeconomic shock.
%
% So we will have two shocks, the macroeconomic shock z2 will have 'its
% own' transition matrix, and then the microeconomic shock z1 will have
% a transition probability that depends on z2 (which we will write as the
% joint transition probability matrix on z=(z1,z2).
%
% This modelling technique originates with Imrohoroglu (1989) - "Cost of business cycles with indivisibilities and liquidity constraints"
% More sophisticated versions of this concept would be "rational inattention" so that households do not react (much) to the macroecomic variables. See,
% MaćkowiakWiederholt (2015) - “Business Cycle Dynamics under Rational Inattention."
% Or as life-cycle model in Carroll, Crawley, Slacalek, Tokuoka & White (2020) - Sticky Expectations and Consumption Dynamics"
% (sticky expectations is a reduced-form way to model the concept of rational inattention) 


%% How does VFI Toolkit think about this?
%
% One decision variable: h, labour hours worked
% One endogenous state variable: a, assets (total household savings)
% Two exogenous state variable: z1 and z2, each of which takes values of 0 or 1 (explained above)
% Age: j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=51; % Endogenous labour choice (fraction of time worked)
n_a=201; % Endogenous asset holdings
n_z=[2,2]; % Exogenous 'unemployment' shock
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

%% Dealing with two markovs: grid and transition matrix, z_grid and pi_z

% We have two independent grids, so can just go back to using a stacked column vector for z_grid
z1_grid=[1;0]; % the first entry is employment and the second is unemployment.
z2_grid=[1;0]; % the first entry is expansion, the second is recession.
z_grid=[z1_grid;z2_grid];

% We will take a few steps to construct pi_z.
% The probabilites of transitioning between a expansion and a recession,
% for z2, are given by
pi_z2=[0.9375,0.0625; 0.0625,0.9375];
% Now, in the recession, we want the transition between employment and unemployment to be:
pi_z1_r=[0.94,0.06; 0.43,0.57];
% While in an expansion,
pi_z1_e=[0.975, 0.025; 0.60, 0.40];

% Then we put this together to get pi_z
% first, just create a matrix of the appropriate size
pi_z=zeros(prod(n_z),prod(n_z)); 
% The interpretation of our grid on z is
% [e,b;
%  u,b;
%  e,r;
%  u,r];
% where e=employment, u=unemployment, b=boom (expansion), r=recession.

% So the first two rows of pi_z are e,u if we are in a boom today
% And first two columns are e,u if we are in a boom tomorrow
pi_z(1:2,1:2)=pi_z1_e*pi_z2(1,1); % employment probabilities given in boom, times probability of transitioning boom-boom
% And last two columns are e,u if we are in recession tomorrow
pi_z(1:2,3:4)=pi_z1_e*pi_z2(1,2);% employment probabilities given in boom, times probability of transitioning boom-recession

% And the last two rows of pi_z are e,u if we are in a recession today
% And first two columns are e,u if we are in a boom tomorrow
pi_z(3:4,1:2)=pi_z1_r*pi_z2(2,1); % employment probabilities given in recession, times probability of transitioning recession-boom
% And last two columns are e,u if we are in recession tomorrow
pi_z(3:4,3:4)=pi_z1_r*pi_z2(2,2);% employment probabilities given in recession, times probability of transitioning recession-recession

% Done, pi_z is now created. We have transition probabilites for z2 which
% are independent of z1. While the transition probabilities for z1 depend on
% the current value of z2.

% Now that we have finished creating z_grid and pi_z, note that we just
% have z_grid as a stacked column vector (as the grids for z1 and z2 are
% independent of each other) while pi_z is, as ever,
% size(pi_z)=prod(n_z)-by-prod(n_z). pi_z can contain anything from very
% independent to very complicated interactions in the transition
% probabilities for z1 and z2 without ever changing shape.

%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

% Use 'LifeCycleModelA5_ReturnFn', which has two markov exogenous states
ReturnFn=@(h,aprime,a,z1,z2,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)...
    LifeCycleModelA5_ReturnFn(h,aprime,a,z1,z2,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
vfoptions=struct(); % Just using the defaults.
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

%% Now, we want to graph Life-Cycle Profiles

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are
% at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,floor((n_z(1)+1)/2),floor((n_z(2)+1)/2))=1; % All agents start with zero assets, and the median shock

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

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (h,aprime,a,z) as first
% inputs, then just any relevant parameters.
FnsToEvaluate.fractiontimeworked=@(h,aprime,a,z1,z2) h; % h is fraction of time worked
FnsToEvaluate.earnings=@(h,aprime,a,z1,z2,w,kappa_j) w*kappa_j*z1*h; % w*kappa_j*z1*h is the labor earnings (note: h will be zero when z is zero, so could just use w*kappa_j*h)
FnsToEvaluate.assets=@(h,aprime,a,z1,z2) a; % a is the current asset holdings

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



