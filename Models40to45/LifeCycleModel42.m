%% Life-Cycle Model 42: experienceassetu (Uncertain Human Capital)
% 'experienceassetu' is when aprime(d,a,u)  --- when next period endogenous
% state cannot be chosen directly, but is instead a function of a decision
% variable, this period endogenous state, and a between-period i.i.d. shock.
% 'experienceassetu' is often used for uncertain human capital models, and in this
% example it will be human capital that you invest time into.

% This model has two endogenous states. VFI Toolkit requires that the
% experienceassetu is the later of the two.

% Because next period endogenous state for the experienceasset is not
% chosen directly, it does not get included as an input to ReturnFn nor
% FnsToEvaluate.

% We have to set up n_a and a_grid appropriately, and then the main thing
% is creating aprimeFn (roughly, lines 105-125).

% The model has two decision variables, when using an experienceassetu with
% decision variables, the later decision variable will be the one the
% toolkit uses for the experienceassetu in aprime(d,a,u).


%% How does VFI Toolkit think about this?
%
% Two decision variable: l, labour hours worked, and s, study time
% Two endogenous state variable: a, assets (total household savings), and h human capital
% One stochastic exogenous state variable: z, an AR(1) process (in logs), idiosyncratic shock to labor productivity per-human-capital units
% One stochastic exogenous between-period i.i.d variable: u, makes the human capital accumulation uncertain
% Age: j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=[11,5]; % Endogenous labour suppy, study-time
n_a=[201,21]; % Endogenous asset holdings, human capital
n_z=11; % Exogenous labor productivity units shock
n_u=5; % Exogenous between-period i.i.d shock
N_j=Params.J; % Number of periods in finite horizon

%% Parameters

% Discount rate
Params.beta = 0.96;
% Preferences
Params.sigma = 2; % Coeff of relative risk aversion (curvature of consumption)
Params.eta = 1.5; % Curvature of leisure
Params.psi = 10; % Weight on leisure
% Note: psi and eta are not exactly the same here as in previous models

% Prices
Params.w=1; % Wage
Params.r=0.05; % Interest rate (0.05 is 5%)

% Demographics
Params.agej=1:1:Params.J; % Is a vector of all the agej: 1,2,3,...,J
Params.Jr=46;

% Pensions
Params.pension=0.1;

% Uncertain Human Capital
Params.alpha_h=0.6; % returns to scale of human capital production fn
Params.delta_h=0.02; % depreciation rate of human capital
Params.ability=3; % ability to produce human capital
Params.sscaler=5; % scale from units of study-time into something more appropriate for the human capital production function
% i.i.d. shocks that make human capital uncertain
Params.sigma_epsilon_u=0.01;


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
asset_grid=10*(linspace(0,1,n_a(1)).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.

h_grid=linspace(1,4,n_a(2))'; % Because h is an experienceasset, it will be interpolated onto this grid and so we need less grid points than usual
% Note: deliberately omit 0 from h_grid

% First, the AR(1) process z
[z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_epsilon_z,n_z);
z_grid=exp(z_grid); % Take exponential of the grid
[mean_z,~,~,~]=MarkovChainMoments(z_grid,pi_z); % Calculate the mean of the grid so as can normalise it
z_grid=z_grid./mean_z; % Normalise the grid on z (so that the mean of z is exactly 1)

% Grid for labour supply choice
l_grid=linspace(0,1,n_d(1))';
% Grid for study-time choice
s_grid=linspace(0,0.5,n_d(2))';

% Put labor supply and study-time grids together
d_grid=[l_grid; s_grid]; % 'stacked-column vector', so one column vector on top of the other
% Put asset and human capital grids together
a_grid=[asset_grid; h_grid]; % 'stacked-column vector', so one column vector on top of the other


%% experienceassetu: aprimeFn
% To use an experienceassetu, we need to define aprime(d,a,u) 
% [in notation of the current model, hprime(s,h,u)]

vfoptions.experienceassetu=1; % Using an experience-asset-u
% Note: by default, assumes it is the last d variable that controls the
% evolution of the experience asset (and that the last a variable is
% the experience asset).

% aprimeFn gives the value of hprime
vfoptions.aprimeFn=@(s,h,u,alpha_h, delta_h, ability,sscaler) u*(ability*(h*(sscaler*s))^alpha_h+h*(1-delta_h));
% The first three inputs must be (d,a,u) [in the sense of aprime(d,a,u)], then any parameters

% We also need to tell simoptions about the experienceassetu
simoptions.experienceassetu=1;
simoptions.aprimeFn=vfoptions.aprimeFn;
simoptions.d_grid=d_grid; % Needed to handle aprimeFn 
simoptions.a_grid=a_grid; % Needed to handle aprimeFn

% And we need to define the i.i.d. shocks u
[u_grid,pi_u]=discretizeAR1_FarmerToda(0,0,Params.sigma_epsilon_u,n_u);
u_grid=exp(u_grid); % Take exponential of the grid
pi_u=pi_u(1,:)'; % as i.i.d. (so stationary dist is just a row of the transition matrix that we created for a markov)
mean_u=sum(pi_u.*u_grid);
u_grid=u_grid./mean_u; % Normalise the grid on u (so that the mean of u is exactly 1)

% We need to store u
vfoptions.n_u=n_u;
vfoptions.u_grid=u_grid;
vfoptions.pi_u=pi_u;
simoptions.n_u=vfoptions.n_u;
simoptions.u_grid=vfoptions.u_grid;
simoptions.pi_u=vfoptions.pi_u;

% To better understand the human capital production function, here is a graph of it
hprime_shu=zeros([n_a,n_u]);
for hh=1:n_a(2)
    for ss=1:n_d(2)
        for uu=1:n_u
            hprime_shu(ss,hh,uu)=vfoptions.aprimeFn(s_grid(ss),h_grid(hh),u_grid(uu),Params.alpha_h,Params.delta_h,Params.ability,Params.sscaler);
        end
    end
end
figure(1)
subplot(3,1,1); plot(h_grid,h_grid,h_grid,hprime_shu(1,:,1),h_grid,hprime_shu(1,:,3),h_grid,hprime_shu(1,:,5))
legend('45 degree','small u', 'mid u', 'large u')
title('min s')
subplot(3,1,2); plot(h_grid,h_grid,h_grid,hprime_shu(3,:,1),h_grid,hprime_shu(3,:,3),h_grid,hprime_shu(3,:,5))
legend('45 degree','small u', 'mid u', 'large u')
title('mid s')
subplot(3,1,3); plot(h_grid,h_grid,h_grid,hprime_shu(5,:,1),h_grid,hprime_shu(4,:,3),h_grid,hprime_shu(5,:,5))
legend('45 degree','small u', 'mid u', 'large u')
title('max s')
hold off 

%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

% Use 'LifeCycleModel42_ReturnFn'
ReturnFn=@(l,s,aprime,a,h,z,w,sigma,eta,psi,agej,Jr,pension,r,warmglow1,warmglow2,warmglow3,beta,sj)...
    LifeCycleModel42_ReturnFn(l,s,aprime,a,h,z,w,sigma,eta,psi,agej,Jr,pension,r,warmglow1,warmglow2,warmglow3,beta,sj);
% Notice how we have (l,aprime,a,h,z,...)
% Follow same decision-next endo-endo-exo ordering as usual, but because h
% is an experienceassetu, we do not include hprime as it is not chosen
% directly.

%% Now solve the value function iteration problem
disp('Test ValueFnIter')
vfoptions.verbose=1; % give feedback
vfoptions.divideandconquer=1; % exploit monotonicity of first endogenous state (assets)
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
[length(n_d)+length(n_a)-1,n_a,n_z,N_j]
% The n_a,n_z,N_j represent the state on which the decisions/policys
% depend, and there is one decision for each decision variable 'd' and each
% endogenous state variable 'a', minus one because the next period
% experienceasset cannot be chosen directly

%% Let's take a quick look at what we have calculated, namely V and Policy

% Convert the policy function to values (rather than indexes).
% Note that there is one policy for participation (p), and another for next period assets (aprime). 
% Because h is an experienceasset, hprime is not chosen directly so is not in Policy
% Policy(1,:,:,:,:) is h, Policy(2,:,:,:,:) is aprime [as function of (a,h,z,j)]

% Rather than plot lots of outputs, we will just look at two, some
% participation decisions and some aprime decisions. 
% Plots are conditional on median z, mid-point of h_grid
zind=ceil(n_z/2);
hind=ceil(n_a(2)/2);
figure(3)
PolicyVals=PolicyInd2Val_Case1_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions);
subplot(2,1,1); surf(asset_grid*ones(1,Params.J),ones(n_a(1),1)*(1:1:Params.J),reshape(PolicyVals(1,:,hind,zind,:),[n_a(1),Params.J]))
title('Policy function: study decision, median h, median z')
xlabel('Assets (a)')
ylabel('Age j')
zlabel('Female participation decision (p)')
subplot(2,1,2); surf(asset_grid*ones(1,Params.J),ones(n_a(1),1)*(1:1:Params.J),reshape(PolicyVals(2,:,hind,zind,:),[n_a(1),Params.J]))
title('Policy function: next period assets (aprime), median h, median z')
xlabel('Assets (a)')
ylabel('Age j')
zlabel('Next period assets (aprime)')

% Because h is an experienceassetu, hprime is not chosen directly so is not in Policy
% But then how does hprime evolve? Remember that hprime(p,h,u), so you can
% use the current h, together with s (which is in Policy) and a value for shock u, and 
% you would then need to pass these as inputs into aprimeFn, which outputs the value
% of hprime(p,h,u). We won't attempt to do that here, but that is what the
% toolkit is doing internally (as well as then using linear interpolation
% to put the value of hprime(p,h,u) back onto the two nearest points on h_grid and deal
% with the different u probabilities/values)


%% Now, we want to graph Life-Cycle Profiles

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are
% at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,4,floor((n_z+1)/2))=1; % All agents start with zero assets, h_grid(4) of human capital, and the median shock [h_grid(4) is roughly same as y_m(1), just my arbitarty decision]

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
% Like with return function, we have to include (l,s,aprime,a,h,z) as first
% inputs, then just any relevant parameters.
FnsToEvaluate.laborsupply=@(l,s,aprime,a,h,z) l; % l is labor supply
FnsToEvaluate.studytime=@(l,s,aprime,a,h,z) s; % s is study-time
FnsToEvaluate.earnings=@(l,s,aprime,a,h,z,w) w*h*z*l; % labor earnings of female (note, p will equal 0 in retirement, so we don't need to treat it separately)
FnsToEvaluate.humancapital=@(l,s,aprime,a,h,z) h; % human capital
FnsToEvaluate.assets=@(l,s,aprime,a,h,z) a; % a is the current asset holdings

% notice that we have called these laborsupply, studytime, earnings, humancapital and assets

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% For example
% AgeConditionalStats.laborsupply.Mean

%% Plot the life cycle profiles of fraction-of-time-worked, earnings, and assets

figure(5)
subplot(5,1,1); plot(Params.agejshifter+(1:1:Params.J),AgeConditionalStats.laborsupply.Mean)
title('Life Cycle Profile: Labor Supply (l)')
subplot(5,1,2); plot(Params.agejshifter+(1:1:Params.J),AgeConditionalStats.studytime.Mean)
title('Life Cycle Profile: Study time (s)')
subplot(5,1,3); plot(Params.agejshifter+(1:1:Params.J),AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Earnings (w h z l)')
subplot(5,1,4); plot(Params.agejshifter+(1:1:Params.J),AgeConditionalStats.humancapital.Mean)
title('Life Cycle Profile: Human Capital (h)')
subplot(5,1,5); plot(Params.agejshifter+(1:1:Params.J),AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Assets (a)')
% Because the calibration is a bit silly, assets are an inferior investment
% to human capital. Hence throughout working life households just invest in
% human capital, and only once they get close to retirement do they instead
% switch it all over to assets.
