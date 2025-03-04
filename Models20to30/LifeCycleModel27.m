%% Life-Cycle Model 27: Earnings Dynamics (Gaussian-Mixtures)
% Solve an exogenous earnings Life-Cycle Model with earnings following the
% process of Model 5 of Guvenen, Karahan, Ozkan & Song (2021). The earnings
% are given by w*(1-upsilon)*exp(kappa_j + alpha_i + z1 + e), where upsilon
% is a binary-valued 'non-employment' shock, kappa_j is a deterministic
% (quadratic) function of age, alpha_i is a fixed-effect distributed across
% households as normal distribution, z1 is an AR(1) with gaussian-mixture
% innovations, and e is an i.i.d. with gaussian-mixture distribution.

% Gaussian-mixtures are 'mixture' of normal/gaussian distributions (here
% two, but can be more). It can be useful to think of a gaussian-mixture of
% two normal distributions as happening in two stages (this is not
% actually true as the gaussian-mixture is a single pdf/cdf, but is a useful 
% thought), in the first stage we draw a probability p, which tells use that 
% with p we are in the first of two normal distributions (that make up our 
% mixture) and with 1-p we are in the second of the two normal distriubtions. 
% Then in the second stage we draw from that normal distribution.

%% How does VFI Toolkit think about this?
%
% Zero decision variables
% One endogenous state variable: a, assets (total household savings)
% Two markov stochastic exogenous state variables: 
%     z1: persistent shocks to labor efficiency, AR(1) with gaussian-mixture innovations
%     upsilon: non-employment shock, whose transition probabilities depend on z1
% One i.i.d. stochastic exogenous state variable: e, transitory shocks to labor efficiency units (which is an i.i.d. gaussian mixture)
% Age: j
% Permanent type: alpha_i, a fixed effect in labor efficiency, distributed across households as a normal distribution

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 25 to age 100, so 76 periods

Params.agejshifter=24; % Age 25 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =76, Number of period in life-cycle

% Grid sizes to use
n_d=0; % This is an exogenous labor supply model
n_a=751; % Endogenous asset holdings
n_z=[17,2]; % Markov exogenous state: non-employment, and z the AR(1) with gaussian-mixture innovations
n_e=9; % i.i.d. which has gaussian-mixture distribution
N_j=Params.J; % Number of periods in finite horizon

% Permanent types
N_i=5;

%% Parameters

% Discount rate
Params.beta = 0.98;
% Preferences
Params.sigma = 2; % Coeff of relative risk aversion (curvature of consumption)

% Prices
Params.w=731; % Wage
Params.r=0.05; % Interest rate (0.05 is 5%)

% Demographics
Params.agej=1:1:Params.J; % Is a vector of all the agej: 1,2,3,...,J
Params.Jr=66-Params.agejshifter; % Age 65 is last working age, age 66 is retired

% Set up taxes, income floor and pensions
% IncomeTax=tau1+tau2*log(Income)*Income, where $IncomeTax$ is the amount paid by a household with $Income$.
% This functional form is found to have a good emprical fit to the US income tax system by GunerKaygusuvVentura2014.
Params.tau1=0.099;
Params.tau2=0.035;
% Pensions
Params.pension=15400;
% Income floor (income in non-employment state during working age)
Params.incomefloor=6400; % Just an initial guess

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
Params.sj=1-Params.dj(25:100); % Conditional survival probabilities
Params.sj(end)=0; % In the present model the last period (j=J) value of sj is actually irrelevant

% Warm glow of bequest
Params.Jbeq=N_j; % Age from which warm-glow of bequests is received
Params.wg1=4; % (relative) importance of bequests
Params.wg2=6.8; % controls how much bequests are viewed as a 'luxury good'
Params.wg3=Params.sigma; % By using the same curvature as the utility of consumption it makes it much easier to guess appropraite parameter values for the warm glow

%% Earnings process: parameters and discretization
% Model 5 of  Guvenen, Karahan, Ozkan & Song (2021). All parameter values are taken from there.

% Age-dependent labor productivity units
Params.kappa_j=2.746+0.624*(Params.agej/10)-0.167*(Params.agej/10).^2;
% Note: agej=age-24, which is what GKOS2021 use. [GKOS2021 normalize agej by 10 so that all the coefficients have similar magnitudes]
Params.kappa_j(Params.Jr:end)=0; % Now fill in the rest of the retirement ages with zero productivity

% z1: AR(1) process with gaussian-mixture innovations, and initial normal distribution in 'period 0'  
Params.rho=0.991*ones(1,Params.Jr-1);
Params.mixprobs_eta=[0.176;1-0.176].*ones(1,Params.Jr-1);
% Gaussian-mixture innovations
Params.mu_eta1=-0.524*ones(1,Params.Jr-1);
% Note mew_eta2 is set to make mean(eta)=0 (from mew_et1 and mixture-probabilities)
kfttoptions.setmixturemutoenforcezeromean=1; % If missing mean for one of the gaussian mixture innovations this is used to set it so that mean equals zero (assumes the missing one is the 'last' one)
Params.sigma_eta1=0.113*ones(1,Params.Jr-1);
Params.sigma_eta2=0.046*ones(1,Params.Jr-1);
% Initial agej=0 distribution of the life-cycle AR(1) process with gaussian-mixture innovations
Params.sigma_z0=0.450; % N(0,sigma_z0)
kfttoptions.initialj0sigma_z=Params.sigma_z0;
% Now we can set up grids and transtition probabilities for all ages
Params.sigma_eta=[Params.sigma_eta1;Params.sigma_eta2];
Params.mu_eta=Params.mu_eta1; % Using kirkbyoptions.setmixturemutoenforcezeromean=1
% Discretize using KFTT (Kirkby-Farmer-Tanaka-Toda, is an extension of Tanaka-Toda to life-cycle AR(1) processes)
% Notice we use KFTT command for a 'LifeCycleAR1wGM', the 'wGM' is 'with gaussian-mixture innovations'
kfttoptions.nSigmas=2; % number of standard deviations for the max and min points of grid used for z1
fprintf('Discretizing z1 using KFTT quadrature method (discretizeLifeCycleAR1wGM_KFTT) \n')
[z1_grid_J, pi_z1_J,jequaloneDistz1,otheroutputs_z1] = discretizeLifeCycleAR1wGM_KFTT(0,Params.rho,Params.mixprobs_eta,Params.mu_eta,Params.sigma_eta,n_z(1),Params.Jr-1,kfttoptions); % z_grid_J is n_z(1)-by-J, so z_grid_J(:,j) is the grid for age j
% pi_z1_J is n_z(1)-by-n_z(1)-by-J, so pi_z1_J(:,:,j) is the transition matrix for age j

% upsilon: non-employment process, follows markov process, whose transition probabilities also depend on z
Params.lambda=0.016;
Params.xi=@(agej,z) -2.495-1.037*(agej/10)-5.051*z-1.087*(agej/10)*z;
Params.prob_upsilon=@(xi) exp(xi)/(1+exp(xi));
% Need two states, 0 and min(1,exp(lambda))
% Note that the probabilities, xi, are based on agej and z in the same period (which is next period in the transition matrix)
% Create grid
upsilon_grid_J=[zeros(1,Params.Jr-1); min(1,exp(Params.lambda))*ones(1,Params.Jr-1)];
% pi_upsilon_J cannot be defined independent of pi_z_J, so create the joint transition matrix for (z, upsilon)
pi_z_J=zeros(n_z(1)*2,n_z(1)*2,Params.Jr-1); % pi_z_J for working ages
for jj=1:Params.Jr-2
    for z1_c=1:n_z(1)
        xi=Params.xi(jj+1,z1_grid_J(z1_c,jj+1));
        prob_upsilon=Params.prob_upsilon(xi);
        % Note all that matters for (next period) upsilon is next period z and next period age
        pi_z_J(1:n_z(1),z1_c,jj)=pi_z1_J(:,z1_c,jj)*(1-prob_upsilon); % Corresponds to upsilon=0
        pi_z_J(n_z(1)+1:2*n_z(1),z1_c,jj)=pi_z1_J(:,z1_c,jj)*(1-prob_upsilon);  % Corresponds to upsilon=0
        pi_z_J(1:n_z(1),n_z(1)+z1_c,jj)=pi_z1_J(:,z1_c,jj)*prob_upsilon; % Corresponds to upsilon=min(1,exp(lambda))
        pi_z_J(n_z(1)+1:2*n_z(1),n_z(1)+z1_c,jj)=pi_z1_J(:,z1_c,jj)*prob_upsilon; % Corresponds to upsilon=min(1,exp(lambda))
    end
end
pi_z_J(:,:,Params.Jr-1)=ones(n_z(1)*2,n_z(1)*2)/(n_z(1)*2); % Note that the agej=Jr-1 transition is irrelevant in any case

% e: i.i.d. with gaussian-mixture distribution
% This can be done using Farmer-Toda method
Params.p_e=[0.044;1-0.044];
Params.mu_e1=0.134;
Params.mu_e2=-(Params.p_e(1)*Params.mu_e1)/Params.p_e(2); % Rearrange: p.*mu=0
Params.mu_e=[Params.mu_e1;Params.mu_e2];
Params.sigma_e1=0.762;
Params.sigma_e2=0.055;
Params.sigma_e=[Params.sigma_e1;Params.sigma_e2];
% Notice we use Farmer-Toda command 'AR1wGM', the 'wGM' is 'with gaussian-mixture innovations'
farmertodaoptions.nSigmas=2; % number of standard deviations for the max and min points of grid used for z
farmertodaoptions.method='GMQ'; % grid points specifically for gaussian-mixture distribution
fprintf('Discretizing e using Farmer-Toda quadrature method (discretizeAR1wGM_FarmerToda) \n')
[e_grid,pi_e, otheroutputs_e] = discretizeAR1wGM_FarmerToda(0,0,Params.p_e,Params.mu_e,Params.sigma_e,n_e,farmertodaoptions);
% Note that e is iid, so
pi_e=pi_e(1,:)';

% Fill in the rest of the retirement ages with zero productivity
z1_grid_J=[z1_grid_J,zeros(n_z(1),Params.J-Params.Jr+1)];
upsilon_grid_J=[upsilon_grid_J,zeros(2,Params.J-Params.Jr+1)];
e_grid_J=[e_grid.*ones(1,Params.Jr-1),zeros(n_e,Params.J-Params.Jr+1)];
pi_e_J=pi_e*ones(1,Params.J);
% Put z1 and upsilon together to get z
z_grid_J=[z1_grid_J; upsilon_grid_J];
% Fill in the retirement ages with uniform transition probabilities (these are anyway irrelevant)
temp=pi_z_J;
pi_z_J=ones(n_z(1)*2,n_z(1)*2,Params.J)/(n_z(1)*2);
pi_z_J(:,:,1:Params.Jr-1)=temp;
% Fill in the retirement ages for z with identity matrix (these are anyway irrelevant)
pi_z_J2=repmat(eye(n_z(1)*2,n_z(1)*2),1,1,Params.J-Params.Jr+1);
pi_z_J(:,:,Params.Jr:Params.J)=pi_z_J2;

% In this model, we also discretize the parameter alpha_i across permanent
alphaoptions.nSigmas=1.2; % number of standard deviations for the max and min points of grid used for alpha
Params.sigma_alpha=0.472;
fprintf('Discretizing alpha using Farmer-Toda quadrature method (discretizeAR1_FarmerToda) \n')
[alpha_grid,pi_alpha] = discretizeAR1_FarmerToda(0,0,Params.sigma_alpha,N_i,alphaoptions);
statdist_alpha=pi_alpha(1,:)'; % Because it is i.i.d., we can just take any row
Params.alpha=alpha_grid; % Store the values of alpha_i in Params
% While we are here, put the distribution over permanent types into Params
% as well and set up 
Params.statdist_alpha=statdist_alpha;
PTypeDistParamNames={'statdist_alpha'};

% Note: in this model, we have exp() of the shocks inside the ReturnFn,
% whereas in most of the earlier models we took the exponential of the grid
% for the shocks and then could just use the shock directly (without exp()) 
% in the ReturnFn.

%% Grids
% The ^3 means that there are more points near 0 and near 10. We know from
% theory that the value function will be more 'curved' near zero assets,
% and putting more points near curvature (where the derivative changes the most) increases accuracy of results.
a_grid=(10^6)*(linspace(0,1,n_a).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.

% To use e variables we have to put them into the vfoptions and simoptions
vfoptions.n_e=n_e;
vfoptions.e_grid=e_grid;
vfoptions.pi_e=pi_e;
simoptions.n_e=vfoptions.n_e;
simoptions.e_grid=vfoptions.e_grid;
simoptions.pi_e=vfoptions.pi_e;

% Grid for labour choice
h_grid=linspace(0,1,n_d)'; % Notice that it is imposing the 0<=h<=1 condition implicitly
% Switch into toolkit notation
d_grid=h_grid;


%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

% Use 'LifeCycleModel27_ReturnFn', and now input z, upsilon and e.
ReturnFn=@(aprime,a,z1,upsilon,e,alpha,w,sigma,agej,Jr,pension,incomefloor,r,kappa_j,wg1,wg2,wg3,beta,sj,tau1,tau2,Jbeq)...
    LifeCycleModel27_ReturnFn(aprime,a,z1,upsilon,e,alpha,w,sigma,agej,Jr,pension,incomefloor,r,kappa_j,wg1,wg2,wg3,beta,sj,tau1,tau2,Jbeq)
% We have no decision variable, one standard endogenous state, two markovs
% and an i.i.d, so inputs start with (aprime,a,upsilon,z,e,...)

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
vfoptions.divideandconquer=1; % exploit monotonicity
disp('Solving ValueFnIter')
tic;
[V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j, N_i, d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Params, DiscountFactorParamNames, vfoptions);
toc

% V is a structure, with N_i fields.
% V.ptype001 is now (a,z1,upsilon,e,j). One dimension for each state variable.
% Compare
size(V.ptype001)
% with
[n_a,n_z,n_e,N_j] % note: n_z is z1 and upsilon
% they are the same.
% Policy is similarly a structure, and
size(Policy.ptype001)
% which is the same as
[length(n_a),n_a,n_z,n_e,N_j] % note: n_z is z1 and upsilon
% The n_a,n_z,n_e,N_j represent the state on which the decisions/policys
% depend, and the only action is to choose aprime.

%% We won't plot the Value and Policy fn, but thinking out how you would might be a good way to check you understand the form of V and Policy

%% Now, we want to graph Life-Cycle Profiles

%% Initial distribution of agents at birth (j=1)

% First, sort out the initial distribution over z1 and upsilon
% Use the period 1 dist of z (from the discretization) with the implied stationary dist of upsilon (conditional on z)
jequaloneDistz=zeros(n_z(1)*n_z(2),1);
for z1_c=1:n_z(1)
    xi=Params.xi(1,z1_grid_J(z1_c,1));
    prob_upsilon=Params.prob_upsilon(xi);
    % Note all that matters for (next period) upsilon is next period z and next period age
    jequaloneDistz(z1_c)=jequaloneDistz1(z1_c)*(1-prob_upsilon); % Corresponds to upsilon=0
    jequaloneDistz(n_z(1)+z1_c)=jequaloneDistz1(z1_c)*prob_upsilon; % Corresponds to upsilon=min(1,exp(lambda))
end
% Now, combine with e: distribute based on dist of e
jequaloneDistze=zeros(n_z(1)*n_z(2)*n_e,1);
for e_c=1:n_e
    jequaloneDistze((1:1:n_z(1)*n_z(2))+(n_z(1)*n_z(2))*(e_c-1))=jequaloneDistz*pi_e(e_c);
end
jequaloneDistze=reshape(jequaloneDistze,[n_z,n_e]);

% We will give them all zero assets.
jequaloneDist=zeros([n_a,n_z,n_e],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,:,:,:)=shiftdim(jequaloneDistze,-1); % All agents start with zero assets, and the distribution over shocks we created just above

% Note: We don't need to put the distribution over permanent types into
% this, as we already have that from PTypeDistParamNames. If you want you
% can put the distribution over permanent types into the initial
% distribution and the codes still work fine.

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
StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightsParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,N_i,pi_z_J,Params,simoptions);

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (aprime,a,z1,upsilon,e) as first
% inputs, then just any relevant parameters.
FnsToEvaluate.earnings=@(aprime,a,z1,upsilon,e,agej,Jr,kappa_j,w,alpha) (agej<Jr)*w*(1-upsilon)*exp(kappa_j+alpha+z1+e); % w*kappa_j*h*z*e is the labor earnings
FnsToEvaluate.assets=@(aprime,a,z1,upsilon,e) a; % h is fraction of time worked
FnsToEvaluate.nonemployment=@(aprime,a,z1,upsilon,e,agej,Jr) (agej<Jr)*upsilon; % upsilon=1 is non-employment
% notice that we have called these earnings, assets and nonemployment

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate,Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,z_grid_J,simoptions);

% For example
% AgeConditionalStats.earnings.Mean
% There are things other than Mean, but in our current deterministic model
% in which all agents are born identical the rest are meaningless.

%% Plot the life cycle profiles of fraction-of-time-worked, earnings, and assets

% Plot the age-conditional means for the population as a whole
figure(1)
subplot(3,1,1); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Labor Earnings [w*(1-upsilon)*exp(kappa_j+alpha+z1+e)]')
subplot(3,1,2); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Assets (a)')
subplot(3,1,3); plot(1:1:Params.J,AgeConditionalStats.nonemployment.Mean)
title('Life Cycle Profile: Fraction in Non-employment (upsilon=1)')

% Plot the mean earnings conditional on permanent types
figure(2)
plot(1:1:Params.J,AgeConditionalStats.earnings.ptype001.Mean)
hold on
plot(1:1:Params.J,AgeConditionalStats.earnings.ptype002.Mean)
plot(1:1:Params.J,AgeConditionalStats.earnings.ptype003.Mean)
plot(1:1:Params.J,AgeConditionalStats.earnings.ptype004.Mean)
plot(1:1:Params.J,AgeConditionalStats.earnings.ptype005.Mean)
hold off
title('Life Cycle Profile by Permanent Type: Labor Earnings [w*(1-upsilon)*exp(kappa_j+alpha+z1+e)]')
legend('ptype001','ptype002','ptype003','ptype004','ptype005')
% Note, this graph hardcodes N_i=5

%% This earnings process is better than most alternatives at hitting both 
% earnings inequality in the cross-section of earnings, and at earnings 
% inequality in lifetime earnings. So let's produce some stats on both of 
% these just to see how it is done.

% Cross-sectional inequality is easy
simoptions.nquantiles=5; % note, simoptions.nquantiles controls number of quantiles, default is ventiles (20)
AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate,Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,z_grid_J,simoptions);

% No built-in commands for lifetime earnings inequality, so we will
% simulate panel data, and then calculate from that.
simoptions.numbersims=10^4; % number of individuals to simulate panel data for
% Define lifetime earnings as average yearly earnings across ages 25-55.
simoptions.nperiods=31; % so only bother simulating the ages up to 55
SimPanelData=SimPanelValues_FHorz_Case1_PType(jequaloneDist,PTypeDistParamNames,Policy,FnsToEvaluate,Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,z_grid_J,pi_z_J, simoptions);
% Data on lifetime earnings for US can be found in Guvenen, Kaplan, Song & Weidner (2022)
% The following code does same thing with our model simulated earnings as GKSW2022 does with US data.
% First, restrict the sample
earningsmin=1885; % 2013 dollars
% According to US Census: 2013 U.S. median house-hold income was $52,250
scaledearningsmin=earningsmin*(AllStats.earnings.Median/552250); 
% convert earningsmin so that it is appropriate fraction of median earnings
% GKSW2022 include households with earnings over a minimum amount (earningsmin)
KeepIndicator=ones(1,simoptions.numbersims);
for ii=1:simoptions.numbersims
    currentearningssim=SimPanelData.earnings(:,ii);
    count=(SimPanelData.earnings(:,ii)>scaledearningsmin);
    if sum(count)<15 % GKSW2022 drop those without at least 15 observations meeting the earnings minimum
        KeepIndicator(ii)=0; % Drop
    end
end
LifetimeEarningsSample=SimPanelData.earnings(:,logical(KeepIndicator));
% Now, compute the lifetime earnings
LifetimeEarningsSample=sum(LifetimeEarningsSample,1)/31; % 31 is years (ages 25 to 55)
% Now, some inequality measures
LorenzCurve_LifetimeEarnings=LorenzCurve_FromSampleObs(LifetimeEarningsSample);
% GKSW2022, Figure 8, provide the std dev of log, and the interquartile range
stddev_logLifetimeEarnings=std(log(LifetimeEarningsSample));
LifetimeEarningsPercentiles=prctile(LifetimeEarningsSample,[10,25,50,75,90]);
LifetimeEarnings_P75P25ratio=LifetimeEarningsPercentiles(4)/LifetimeEarningsPercentiles(2);
LifetimeEarnings_P90P50ratio=LifetimeEarningsPercentiles(5)/LifetimeEarningsPercentiles(3);
LifetimeEarnings_P50P10ratio=LifetimeEarningsPercentiles(3)/LifetimeEarningsPercentiles(1);


% Print to screen a summary of what found about earnings inequality and lifetime earnings inequality.
% Note, if we want things like the 'share of X percentile', these are
% essentially already all there in the lorenz curves (which has 100 points
% by default, equally spaced)
fprintf('Cross-sectional earnings inequality \n')
fprintf('Gini: %1.2f \n', AllStats.earnings.Gini)
fprintf('Share of 1st (bottom) quintile: %2.2f \n', 100*(AllStats.earnings.LorenzCurve(20)))
fprintf('Share of 2nd quintile: %2.2f \n', 100*(AllStats.earnings.LorenzCurve(40)-AllStats.earnings.LorenzCurve(20)))
fprintf('Share of 3rd quintile: %2.2f \n', 100*(AllStats.earnings.LorenzCurve(60)-AllStats.earnings.LorenzCurve(40)))
fprintf('Share of 4th quintile: %2.2f \n', 100*(AllStats.earnings.LorenzCurve(80)-AllStats.earnings.LorenzCurve(60)))
fprintf('Share of 5th (top) quintile: %2.2f \n', 100*(1-AllStats.earnings.LorenzCurve(80)))
fprintf('Share of 90-94th percentiles: %2.2f \n', 100*(AllStats.earnings.LorenzCurve(94)-AllStats.earnings.LorenzCurve(89)))
fprintf('Share of 95-99th percentiles: %2.2f \n', 100*(AllStats.earnings.LorenzCurve(99)-AllStats.earnings.LorenzCurve(94)))
fprintf('Share of Top 1 percentile: %2.2f \n', 100*(1-AllStats.earnings.LorenzCurve(99)))

fprintf('Lifetime earnings inequality \n')
fprintf('Std dev of log: %1.2f \n', stddev_logLifetimeEarnings)
fprintf('Interquartile range %1.2f \n', LifetimeEarnings_P75P25ratio)
fprintf('P90/P50 ratio: %1.2f \n', LifetimeEarnings_P90P50ratio)
fprintf('P50/P10 range: %1.2f \n', LifetimeEarnings_P50P10ratio)


