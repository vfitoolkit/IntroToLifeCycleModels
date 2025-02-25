% Adds fertility and children to 'Life Cycle Model 9' from 'Introduction to Life-Cycle Models'
%
% Uses what VFI Toolkit call a "semi-exogenous state". An exogenous state that can be influenced by the decision variables.
% The changes to life-cycle model 9 involve adding a decision variable for 'fertility' which influences the two semi-exogenous 
% states 'number of infants', n1, and 'number of children', n2.
% A household that chooses to try and have a child (fertility=1) then has a probability (probofbirth) of having an infant (n1=1) next period.
% Infants age stochastically into children, and children age stochastically into adults (at which point they disappear from the present household
% problem). The use of stochastic aging is a trick to reduce the size of the state space of the problem.
%
% At the bottom of this code you can see pi_semiz_J which is what the codes create internally to handle the transitions of the semi-exogenous state
% that depend on the decision (d) variable, and potentially on age (j). The dependence on age j in the current model comes because probofbirth is an
% age-dependent parameter.

%% How does VFI Toolkit think about this?
%
% Two decision variable: h and f, labour hours worked and fertility decision
% One endogenous state variable: a, assets (total household savings)
% One stochastic exogenous state variable: z, an AR(1) process (in logs), idiosyncratic shock to labor productivity units
% Age: j
%
% 'Last' decision variable influences the semi-exogenous states: f, fertility decision
% One semi-exogenous state: infants (number of)
% One semi-exogenous state: children (number of)

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=[51,2]; % Endogenous labour choice (fraction of time worked), fertility decision
n_a=201; % Endogenous asset holdings
n_semiz=[2,4]; %number of infants, number of children
n_z=11; %Exogenous labor productivity units shock
N_j=Params.J; % Number of periods in finite horizon

%% Additional parameters specific to fertility
Params.Jf=20; % Not actually used for anything, what matters for code is just the probability of birth and this is zero after the Jf-th period

Params.probofbirth=[0.9020, 0.8857, 0.8708, 0.8569, 0.8435, 0.8302, 0.8166, 0.8024, 0.7869, 0.7700, 0.7511, 0.7298, 0.7058, 0.6786, 0.6478, 0.6129, 0.5736, 0.5295, 0.4801, 0.4250, zeros(1,Params.J-Params.Jf)]; % lambda_j, fertility opportunities arrives stochastically in each period with probability lambda_j conditional on decision to have child
Params.probofchild=1/4; % theta_c, the probability of an infant becoming a child
Params.probofadult=1/15; % theta_a, the probability of a child becoming an adult

% Utility of children
Params.eta1=40; % Relative weight of children (vs consumption and leisure)
Params.eta2=0.437;
Params.eta3=29-Params.agejshifter; % Roughly the age at which people start wanting to have children
Params.nbar=2.41;

Params.hbar=1; % total time
% Time costs of looking after infants
Params.h_c=0.105; % time cost of infants

% Childcare costs
Params.childcarec=0.5; % Cost of childcare for infants (paid if working)

% Grids for number of infants and number of children
infant_grid=[0;1];
children_grid=(0:1:(n_semiz(2)-1))';

% Set up the semi-exogneous states
vfoptions.n_semiz=n_semiz;
vfoptions.semiz_grid=[infant_grid; children_grid];
% Define the transition probabilities of the semi-exogenous states
vfoptions.SemiExoStateFn=@(n1,n2,n1prime,n2prime,f,probofbirth,probofchild,probofadult) LifeCycleModel29_SemiExoStateFn(n1,n2,n1prime,n2prime,f,probofbirth,probofchild,probofadult);
% It is hardcoded that only the 'last' decision variable can influence the transition probabilities of the semi-exogenous states
% The semi-exogenous states must be included in the return fn, fns to evaluate, etc. The ordering must be that semi-exogenous states come after this period endogenous state(s) and before any markov exogenous states, so (...,a,semiz,z,...)

% We also need to tell simoptions about the semi-exogenous states
simoptions.n_semiz=vfoptions.n_semiz;
simoptions.semiz_grid=vfoptions.semiz_grid;
simoptions.SemiExoStateFn=vfoptions.SemiExoStateFn;

% At the bottom of this code/script there are some lines showing you what
% pi_semiz_J which is created internally looks like.

%% Parameters

% Discount rate
Params.beta = 0.96;
% Preferences
Params.sigma=2; % Coeff of relative risk aversion (curvature of consumption)
Params.eta=1.5; % Curvature of leisure (This will end up being 1/Frisch elasticity)
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

% Grid for labour force participation choice
h_grid=linspace(0,1,n_d(1))'; % Notice that it is imposing the 0<=h<=1 condition implicitly
% Grid for fertility decision
f_grid=[0;1];
% Switch into toolkit notation
d_grid=[h_grid; f_grid];

%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

% Use 'LifeCycleModel29_ReturnFn'
ReturnFn=@(h,f,aprime,a,n1,n2,z,w,sigma,psi,eta,agej,eta1,eta2,eta3,nbar,hbar,h_c,childcarec,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj) ...
    LifeCycleModel29_ReturnFn(h,f,aprime,a,n1,n2,z,w,sigma,psi,eta,agej,eta1,eta2,eta3,nbar,hbar,h_c,childcarec,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj);

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Solve ValueFnIter')
vfoptions.verbose=1;
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

%% Look at the fertility decisions
PolicyVals=PolicyInd2Val_Case1_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions); % Note: n_semiz is in simoptions

figure(1)
% Note: uses max value of z shock
subplot(6,2,1); plot(a_grid,PolicyVals(2,:,1,1,end,1)) % j=1
title('Policy for f at age j=1')
subplot(6,2,3); plot(a_grid,PolicyVals(2,:,1,1,end,5)) % j=5
title('Policy for f at age j=5')
subplot(6,2,5); plot(a_grid,PolicyVals(2,:,1,1,end,10)) % j=10
title('Policy for f at age j=10')
subplot(6,2,7); plot(a_grid,PolicyVals(2,:,1,1,end,15)) % j=15
title('Policy for f at age j=15')
subplot(6,2,9); plot(a_grid,PolicyVals(2,:,1,1,end,20)) % j=20
title('Policy for f at age j=20')
subplot(6,2,11); plot(a_grid,PolicyVals(2,:,1,1,end,25)) % j=25
title('Policy for f at age j=25')
xlabel('Fertility Decision for Household without Children')
subplot(6,2,2); plot(a_grid,PolicyVals(2,:,2,1,end,1)) % j=1
title('Policy for f at age j=1')
subplot(6,2,4); plot(a_grid,PolicyVals(2,:,2,1,end,5)) % j=5
title('Policy for f at age j=5')
subplot(6,2,6); plot(a_grid,PolicyVals(2,:,2,1,end,10)) % j=10
title('Policy for f at age j=10')
subplot(6,2,8); plot(a_grid,PolicyVals(2,:,2,1,end,15)) % j=15
title('Policy for f at age j=15')
subplot(6,2,10); plot(a_grid,PolicyVals(2,:,2,1,end,20)) % j=20
title('Policy for f at age j=20')
subplot(6,2,12); plot(a_grid,PolicyVals(2,:,2,1,end,25)) % j=20
title('Policy for f at age j=25')
xlabel('Fertility Decision for Household with Infant')
% Note that a household with an infant cannot have a second, so they 

% A look at how the utility of children varies with age
figure(2)
plot(exp(Params.agej-Params.eta3)./(1+exp(Params.agej-Params.eta3)))
xlabel('age j')
title('How the utility of children varies with age')
% Note that it is effectively saying no children before agej is close to eta3

%% Now, we want to graph Life-Cycle Profiles

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are
% at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_semiz,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,1,1,floor((n_z+1)/2))=1; % All agents start with zero assets, and the median shock, no infants, no children

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

% Need to also tell simoptions about the semi-exogenous shocks
% Because evaluating pi_semiz_J requires the d_grid we also have to provide
simoptions.d_grid=d_grid;

StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
% Again, we will explain in a later model what the stationary distribution
% is, it is not important for our current goal of graphing the life-cycle profile

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (h,aprime,a,z) as first
% inputs, then just any relevant parameters.
FnsToEvaluate.fractiontimeworked=@(h,f,aprime,a,n1,n2,z) h; % h is fraction of time worked
FnsToEvaluate.earnings=@(h,f,aprime,a,n1,n2,z,w,kappa_j) w*kappa_j*z*h; % w*kappa_j*z*h is the labor earnings (note: h will be zero when z is zero, so could just use w*kappa_j*h)
FnsToEvaluate.assets=@(h,f,aprime,a,n1,n2,z) a; % a is the current asset holdings
FnsToEvaluate.ninfants=@(h,f,aprime,a,n1,n2,z) n1; % a is the current asset holdings
FnsToEvaluate.nchildren=@(h,f,aprime,a,n1,n2,z) n2; % a is the current asset holdings

% notice that we have called these fractiontimeworked, earnings and assets

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% For example
% AgeConditionalStats.earnings.Mean
% There are things other than Mean, but in our current deterministic model
% in which all agents are born identical the rest are meaningless.

%% Plot the life cycle profiles of fraction-of-time-worked, earnings, and assets

figure(5)
subplot(5,1,1); plot(1:1:Params.J,AgeConditionalStats.fractiontimeworked.Mean)
title('Life Cycle Profile: Fraction Time Worked (h)')
subplot(5,1,2); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Labor Earnings (w kappa_j h)')
subplot(5,1,3); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Assets (a)')
subplot(5,1,4); plot(1:1:Params.J,AgeConditionalStats.ninfants.Mean)
title('Life Cycle Profile: Number of Infants (n1)')
subplot(5,1,5); plot(1:1:Params.J,AgeConditionalStats.nchildren.Mean)
title('Life Cycle Profile: Number of children (n2)')

%% If you want to take a look at what the whole 'semi-exogenous transition matrix' looks like (it is created automatically by codes) it will look like

pi_semiz_J=zeros([n_semiz,n_semiz,n_d(2),N_j]); % Note that endogenous state is the first, then the conditional transition matrix for shocks
for f_c=1:n_d(2)
    for n1_c=1:n_semiz(1)
        for n2_c=1:n_semiz(2)
            for n1prime_c=1:n_semiz(1)
                for n2prime_c=1:n_semiz(2)
                    for jj=1:N_j
                        pi_semiz_J(n1_c,n2_c,n1prime_c,n2prime_c,f_c,jj)=vfoptions.SemiExoStateFn(n1_c-1,n2_c-1,n1prime_c-1,n2prime_c-1,f_c-1,Params.probofbirth(jj),Params.probofchild,Params.probofadult); % Note: the -1 turn the index into the value
                    end
                end
            end
        end
    end
end
% Note that pi_semiz_J2 is just treating the two semi-exogenous states as a single vector-valued state.
pi_semiz_J2=reshape(pi_semiz_J,[prod(n_semiz),prod(n_semiz),n_d(2),N_j]);
% Make sure the 'rows' sum to one
for jj=1:N_j
    for f_c=1:n_d(2)
        temp=sum(pi_semiz_J2(:,:,f_c,jj),2);
        if any(abs(temp-1)>10^(-14))
            temp-1
        end
    end
end
% Conditional on the decision variable and age, this just looks like a standard markov transition matrix
pi_semiz_J2(:,:,1,1) % arbitrarily use the first f and j



