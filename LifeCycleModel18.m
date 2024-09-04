%% Life-Cycle Model 18: Precautionary savings with endogenous labor
% We saw precautionary savings with exogenous labor in Life-Cycle Model 17.
% We now repeat this, but this time with endogenous labor.
% Recall that precautionary savings are about increasing asset holdings as a way for households 
% to help reduce the probability of the borrowing constraints binding.
% Now endogenous labor supply means there is another way to avoid reduce the probability of the 
% borrowing constraint binding by working more when assets are low.
% We will see that this 'precautionary labor supply' acts as an substitute for precautionary savings.

% We first solve the model with endogneous labor and then solve again with exogenous labor.

% We can then compare the policy functions and the aggregate assets and labor supply for the models 
% with endogenous vs exogenous labor. We can see how endogenous labor reduces precautionary savings 
% (there are still precautionary savings, which you would see by comparing the stochastic model with 
% endogenous labor to the deterministic model with endogenous labor in the same manner as we did for 
% exogenous labor in Life-Cycle Model 17, just not as much precautionary savings with endogenous labor
% as there were with exogenous labor).

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
n_d=101; % Endogenous labour choice (fraction of time worked)
n_a=501; % Endogenous asset holdings
n_z=3; % Exogenous labor productivity units shock
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
[mean_z,~,~,statdist_z]=MarkovChainMoments(z_grid,pi_z); % Calculate the mean of the grid so as can normalise it
z_grid=z_grid./mean_z; % Normalise the grid on z (so that the mean of z is exactly 1)

% Grid for labour choice
h_grid=linspace(0,1,n_d)'; % Notice that it is imposing the 0<=h<=1 condition implicitly
% Switch into toolkit notation
d_grid=h_grid;

%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

% Notice we still use 'LifeCycleModel8_ReturnFn'
ReturnFn=@(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj) LifeCycleModel8_ReturnFn(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
vfoptions=struct(); % Just using the defaults.
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

%% Now, we want to graph Life-Cycle Profiles

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are
% at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,:)=statdist_z; % All agents start with zero assets, and the median shock

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
% Again, we will explain in a later model what the stationary distribution
% is, it is not important for our current goal of graphing the life-cycle profile

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (h,aprime,a,z) as first
% inputs, then just any relevant parameters.
FnsToEvaluate.fractiontimeworked=@(h,aprime,a,z) h; % h is fraction of time worked
FnsToEvaluate.earnings=@(h,aprime,a,z,w,kappa_j) w*kappa_j*z*h; % w*kappa_j*z*h is the labor earnings (note: h will be zero when z is zero, so could just use w*kappa_j*h)
FnsToEvaluate.assets=@(h,aprime,a,z) a; % a is the current asset holdings
FnsToEvaluate.consumption=@(h,aprime,a,z,agej,Jr,w,kappa_j,r,pension) (agej<Jr)*(w*kappa_j*z*h+(1+r)*a-aprime)+(agej>=Jr)*(pension+(1+r)*a-aprime);

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

%% Before changing model calculate PolicyVals
PolicyVals=PolicyInd2Val_Case1_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions);
% And aggregate variables
AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z,N_j, d_grid, a_grid, z_grid,[],simoptions);

%% Solve endogenous labor supply again, but now with no shocks
store_n_z=n_z; % Need n_z later to do exogenous labor with shocks
n_z=1;
z_grid=1;
pi_z=1;
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,1)=1; 
% ReturnFn is unchanged
[V_noshock, Policy_noshock]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
StationaryDist_noshock=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy_noshock,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
% FnsToEvaluate are unchanged
AgeConditionalStats_noshock=LifeCycleProfiles_FHorz_Case1(StationaryDist_noshock,Policy_noshock,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

PolicyVals_noshock=PolicyInd2Val_Case1_FHorz(Policy_noshock,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions);
AggVars_noshock=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist_noshock, Policy_noshock, FnsToEvaluate, Params, [], n_d, n_a, n_z,N_j, d_grid, a_grid, z_grid,[],simoptions);

%% Parameter to make exogneous labor supply model have the same mean earnings as endogenous labor supply does.
% Households work a fraction of the time. Since the difference in earnings
% between exogenous and endogenous labor supply is just the multiplying by
% hours worked it follows that the endogenous labor supply model will
% have lower mean earnings.
% Note that the ratio is not given by E[h], because of composition effects
% (households with endogenous labor work more/less for certain exogenous
% shocks and at certain ages and asst levels)
%
% We will use a parameter to multiply the earnings with exogenous labor, so as to get the same mean earnings.
Params.meanearningsratio=0.4767/0.9682;
% First I ran this code with meanearningsratio=1, then based on the actual
% resulting ratio of mean earnings (the codes print the mean earnings of
% each model to screen near the end, 0.4767 was mean earnings for model with endogenous labor supply,
% 0.9682 was mean earnings for model with exogenous labor supply) I changed it to the current value.

%% Solve a second time, but this time with exogenous labor supply
% Turn shocks back on
n_z=store_n_z;
[z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho_z,Params.sigma_epsilon_z,n_z);
z_grid=exp(z_grid); % Take exponential of the grid
[mean_z,~,~,statdist_z]=MarkovChainMoments(z_grid,pi_z); % Calculate the mean of the grid so as can normalise it
z_grid=z_grid./mean_z; % Normalise the grid on z (so that the mean of z is exactly 1)
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,:)=statdist_z; % All agents start with zero assets, and the median shock

% Switch to exogenous labor supply
n_d=0; % None
d_grid=[]; % No decision variables
% Change return function
ReturnFn=@(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj,meanearningsratio) LifeCycleModel18B_ReturnFn(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj,meanearningsratio)
[V_exo, Policy_exo]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
StationaryDist_exo=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy_exo,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
% Change FnsToEvaluate
FnsToEvaluate_exo.earnings=@(aprime,a,z,w,kappa_j,meanearningsratio) meanearningsratio*w*kappa_j*z; % z is the 'stochastic endowment' or 'exogenous earnings'
FnsToEvaluate_exo.assets=@(aprime,a,z) a; % a is the current asset holdings
FnsToEvaluate_exo.consumption=@(aprime,a,z,agej,Jr,w,kappa_j,r,pension,meanearningsratio) (agej<Jr)*(meanearningsratio*w*kappa_j*z+(1+r)*a-aprime)+(agej>=Jr)*(pension+(1+r)*a-aprime);

AgeConditionalStats_exo=LifeCycleProfiles_FHorz_Case1(StationaryDist_exo,Policy_exo,FnsToEvaluate_exo,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

PolicyVals_exo=PolicyInd2Val_FHorz_Case1(Policy_exo,n_d,n_a,n_z,N_j,d_grid,a_grid);
AggVars_exo=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist_exo, Policy_exo, FnsToEvaluate_exo, Params, [], n_d, n_a, n_z,N_j, d_grid, a_grid, z_grid,[],simoptions);

%% Solve a third time, this time with exogenous labor supply and no shocks (deterministic model)
n_z=1;
z_grid=1;
pi_z=1;
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,1)=1; 
[V_exonoshock, Policy_exonoshock]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
StationaryDist_exonoshock=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy_exonoshock,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
% FnsToEvaluate_exo are unchangecd
AgeConditionalStats_exonoshock=LifeCycleProfiles_FHorz_Case1(StationaryDist_exonoshock,Policy_exonoshock,FnsToEvaluate_exo,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

PolicyVals_exonoshock=PolicyInd2Val_FHorz_Case1(Policy_exonoshock,n_d,n_a,n_z,N_j,d_grid,a_grid);
AggVars_exonoshock=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist_exonoshock, Policy_exonoshock, FnsToEvaluate_exo, Params, [], n_d, n_a, n_z,N_j, d_grid, a_grid, z_grid,[],simoptions);

%% Plot the savings policies at low asset levels for the three models to see precautionary spending

zind=2; % This will be the median

% Plot the aprime policy as a function (of assets) for a given age  (I do a few for different ages)
figure(1)
subplot(3,1,1); plot(a_grid,PolicyVals(2,:,1,1),a_grid,PolicyVals(2,:,zind,1),a_grid,PolicyVals(2,:,end,1),a_grid,PolicyVals_noshock(1,:,1,1)) % j=1
title('Policy for aprime at age j=1 (at low asset levels)')
legend('endo: min z','endo: median z','endo: max z','endo, no shock')
xlim([0,0.1])
subplot(3,1,2); plot(a_grid,PolicyVals_exo(1,:,1,1),a_grid,PolicyVals_exo(1,:,zind,1),a_grid,PolicyVals_exo(1,:,end,1),a_grid,PolicyVals_exonoshock(1,:,1,1)) % j=1
title('Policy for aprime at age j=1 (at low asset levels)')
legend('exo: min z','exo: median z','exo: max z','exo, no shock')
xlim([0,0.1])
subplot(3,1,3); plot(a_grid,PolicyVals(2,:,zind,1),a_grid,PolicyVals_exo(1,:,zind,1),a_grid,PolicyVals_noshock(1,:,1,1),a_grid,PolicyVals_exonoshock(1,:,1,1)) % j=1
title('Policy for aprime at age j=1 (at low asset levels)')
legend('endo: median z','exo: median z','endo, no shock','exo, no shock')
xlim([0,0.1])

% Plot the aprime policy as a function (of assets) for a given age  (I do a few for different ages)
figure(2)
subplot(3,1,1); plot(a_grid,PolicyVals(2,:,1,20),a_grid,PolicyVals(2,:,zind,20),a_grid,PolicyVals(2,:,end,20),a_grid,PolicyVals_noshock(1,:,1,20)) % j=20
title('Policy for aprime at age j=20 (at low asset levels)')
legend('endo: min z','endo: median z','endo: max z','endo, no shock')
xlim([0,0.1])
subplot(3,1,2); plot(a_grid,PolicyVals_exo(1,:,1,20),a_grid,PolicyVals_exo(1,:,zind,20),a_grid,PolicyVals_exo(1,:,end,20),a_grid,PolicyVals_exonoshock(1,:,1,20)) % j=20
title('Policy for aprime at age j=20 (at low asset levels)')
legend('exo: min z','exo: median z','exo: max z','exo, no shock')
xlim([0,0.1])
subplot(3,1,3); plot(a_grid,PolicyVals(2,:,zind,20),a_grid,PolicyVals_exo(1,:,zind,20),a_grid,PolicyVals_noshock(1,:,1,20),a_grid,PolicyVals_exonoshock(1,:,1,20)) % j=20
title('Policy for aprime at age j=20 (at low asset levels)')
legend('endo: median z','exo: median z','endo, no shock','exo, no shock')
xlim([0,0.1])
% Note: Only younger ages are plotted here. For the reasons explained in Life-Cycle

% Plot the aprime policy as a function (of assets) for a given age  (I do a few for different ages)
figure(3)
subplot(3,1,1); plot(a_grid,PolicyVals(2,:,1,40),a_grid,PolicyVals(2,:,zind,40),a_grid,PolicyVals(2,:,end,40),a_grid,PolicyVals_noshock(1,:,1,40)) % j=20
title('Policy for aprime at age j=40 (at low asset levels)')
legend('endo: min z','endo: median z','endo: max z','endo, no shock')
xlim([0,0.1])
subplot(3,1,2); plot(a_grid,PolicyVals_exo(1,:,1,40),a_grid,PolicyVals_exo(1,:,zind,40),a_grid,PolicyVals_exo(1,:,end,40),a_grid,PolicyVals_exonoshock(1,:,1,40)) % j=20
title('Policy for aprime at age j=40 (at low asset levels)')
legend('exo: min z','exo: median z','exo: max z','exo, no shock')
xlim([0,0.1])
subplot(3,1,3); plot(a_grid,PolicyVals(2,:,zind,40),a_grid,PolicyVals_exo(1,:,zind,40),a_grid,PolicyVals_noshock(1,:,1,40),a_grid,PolicyVals_exonoshock(1,:,1,40)) % j=20
title('Policy for aprime at age j=40 (at low asset levels)')
legend('endo: median z','exo: median z','endo, no shock','exo, no shock')
xlim([0,0.1])


figure(4)
subplot(3,1,1); plot(a_grid,PolicyVals(1,:,1,1),a_grid,PolicyVals(1,:,zind,1),a_grid,PolicyVals(1,:,end,1),a_grid,PolicyVals_noshock(1,:,1,1)) % j=20
title('Policy for h at age j=1 (at low asset levels)')
legend('endo: min z','endo: median z','endo: max z','endo, no shock')
xlim([0,0.1])
subplot(3,1,2); plot(a_grid,PolicyVals(1,:,1,20),a_grid,PolicyVals(1,:,zind,20),a_grid,PolicyVals(1,:,end,20),a_grid,PolicyVals_noshock(1,:,1,20)) % j=20
title('Policy for h at age j=20 (at low asset levels)')
legend('endo: min z','endo: median z','endo: max z','endo, no shock')
xlim([0,0.1])
subplot(3,1,3); plot(a_grid,PolicyVals(1,:,1,40),a_grid,PolicyVals(1,:,zind,40),a_grid,PolicyVals(1,:,end,40),a_grid,PolicyVals_noshock(1,:,1,40)) % j=40
title('Policy for h at age j=40 (at low asset levels)')
legend('endo: min z','endo: median z','endo: max z','endo, no shock')
xlim([0,0.1])
% Using labor supply at low asset levels as an alternative to precautionary labor supply

%%
fprintf(' \n')
fprintf('Endogenous labor supply: \n')
fprintf('   Aggregate labor supply: %8.4f \n',AggVars.fractiontimeworked.Mean)
fprintf('   Aggregate earnings:     %8.4f \n',AggVars.earnings.Mean)
fprintf('   Aggregate assets:       %8.4f \n',AggVars.assets.Mean)
fprintf('   Capital/Income Ratio:   %8.4f \n',AggVars.assets.Mean/AggVars.earnings.Mean)
fprintf('Endogenous labor supply, no shocks: \n')
fprintf('   Aggregate labor supply: %8.4f \n',AggVars_noshock.fractiontimeworked.Mean)
fprintf('   Aggregate earnings:     %8.4f \n',AggVars_noshock.earnings.Mean)
fprintf('   Aggregate assets:       %8.4f \n',AggVars_noshock.assets.Mean)
fprintf('   Capital/Income Ratio:   %8.4f \n',AggVars_noshock.assets.Mean/AggVars_noshock.earnings.Mean)
fprintf('Exogenous labor supply: \n')
fprintf('   Aggregate earnings:     %8.4f \n',AggVars_exo.earnings.Mean)
fprintf('   Aggregate assets:       %8.4f \n',AggVars_exo.assets.Mean)
fprintf('   Capital/Income Ratio:   %8.4f \n',AggVars_exo.assets.Mean/AggVars_exo.earnings.Mean)
fprintf('Exogenous labor supply, no shocks: \n')
fprintf('   Aggregate earnings:     %8.4f \n',AggVars_exonoshock.earnings.Mean)
fprintf('   Aggregate assets:       %8.4f \n',AggVars_exonoshock.assets.Mean)
fprintf('   Capital/Income Ratio:   %8.4f \n',AggVars_exonoshock.assets.Mean/AggVars_exonoshock.earnings.Mean)
fprintf(' \n')
fprintf(' \n')
fprintf('Ratio of earnings of endogenous labor supply model to exogenous labor supply model: %8.4f \n', AggVars.earnings.Mean/AggVars_exo.earnings.Mean)





