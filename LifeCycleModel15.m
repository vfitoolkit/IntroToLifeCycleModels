%% Life-Cycle Model 15: Consumption and Borrowing Constraints 1
% Households would like to consumption smooth
% Because their income is hump-shaped in age (increasing until aroung ages 45-55, when it peaks and 
% then dips slightly until retirement) they would like to shift some of this income to consume when young.
% This means they would like to borrow when young.
% Borrowing constraints (a lower bound on assets) mean that they cannot do this.
% Lets see this in action. We solve the model with a borrowing constraint

% To make the borrowing constraints very obvious we will set the life-cycle
% profile of income, which is determined by kappa_j, to be very steep. And
% we have households born with zero assets (this is what we have typically done anyway).
% We also make the pensions extra generous. All three of these increase the
% amount the households would like to borrow at a young age.

% If you look at the life-cycles you can see how consumption has a kink
% because it is very minimal in the first few years when the budget
% constraint binds, and that this results in a very high marginal utility
% of consumption. A very high marginal utility tells us that anything which
% loosens the borrowing constraint (ability to borrow, a lump-sum
% transfer, etc) would both give a lot of utility to the household and that
% most of it would be consumed (high marginal propensity to consume; how
% high depends on how long borrowing constraint is expected to continue to
% bind).

% How important in borrowing? And how does it change things?
% To make borrowing possible, and hence to be able to see how important it
% is, we need to change the asset grid to allow negative assets. We will
% then enforce the borrowing contraint by using a parameter via the return
% function (Params.borrowingconstraint). Have added some extra points to
% assets cover these negative values (and generally lots of points to make 
% changes in consumption stand out more). Note that we therefore need to modify
% the code to make every household start with zero assets as this is no
% longer the first grid point in a_grid (which is what was done in earlier
% models). Note that giving people lump-sum transfers would have the exact
% same impact as loosening the borrowing constraint.

% You can change Params.borrowingconstraint to make more borrowing
% possible, notice that if you set it so -1 or -5 the consumption gets
% smoother and if you set the borrowing constraint to -10 then consumption
% is essentially perfectly smoothed and looks like a straight line (note that we fix 
% the y-axis on the graphs of the life-cycle profiles for consumption and marginal 
% utility of consumption to make it easier to compare how they look for different borrowing limits).

% I suggest running this code for various values of
% Params.borrowingconstraint to see how the life-cycle profiles of
% consumption and marginal utility of consumption change.
% You could even try flattening the earnings profile, or cutting pensions
% and see how that changed things (think about what you expect to happen
% first, then find out if you are correct ;)

% Comment: The slope of consumption over the life-cycle is determined by
% beta and r (in this model, some additional factor can matter in more
% complex models) and if you chose these appropriately you can make it
% perfectly flat. Specifically beta=1/(1+r) will make consumption perfectly
% flat [You could show this by deriving the 'consumption euler eqn' and
% noting that constant consumption requires beta*(1+r)=1]
% If you want to see this 'flat' consumption 

% Comment: a vaguely related concept is the 'natural borrowing limit' which is the amount 
% that will never bind under the requirement that you die with (next period) assets 
% greater than or equal to zero (typically in the final period). It is the amount that
% you could never repay even if you saved all the earnings in your
% lifetime. You will never borrow that amount, and so it does not have
% similar effects to the borrowing constraints considered above. But if
% there is no borrowing limit full stop then you would choose to consume a
% lot, load up on ever more debt, and die with heaps of debt; imposing a
% 'natural borrowing limit' is simply a way to eliminate this solution (of
% every increasing debt), but has otherwise no effect on the remaining 
% solution. Note that if the model has a borrowing constraint, then it is
% irrelevant if we assume a natural borrowing limit. Note also that if you
% do set a borrowing constaint below the natural borrowing limit, then it
% will have no impact (other than eliminating the solution the natural
% borrowing limit eliminates, that were households borrow ever increasing
% amounts).


%% How does VFI Toolkit think about this?
%
% No decision variable
% One endogenous state variable: a, assets (total household savings)
% No stochastic exogenous state variables
% Age: j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=0; % Endogenous labour choice (fraction of time worked)
n_a=1001; % Endogenous asset holdings
n_z=1; % This is how the VFI Toolkit thinks about deterministic models
N_j=Params.J; % Number of periods in finite horizon

%% Parameters

% Discount rate
Params.beta = 0.96;
% Preferences
Params.sigma = 2; % Coeff of relative risk aversion (curvature of consumption)

% Borrowing constraint
Params.borrowingconstraint=-0; % this number is the minimum amount of assets
% See what happens if you change this to -3, -5 or -10
% You cannot set this to less than -10 (as that is minimum value of the grid on assets)

% Prices
Params.w=1; % Wage
Params.r=0.05; % Interest rate (0.05 is 5%)
% Params.r=1/Params.beta-1; % If you change r to this then the consumption will be perfectly flat

% Demographics
Params.agej=1:1:Params.J; % Is a vector of all the agej: 1,2,3,...,J
Params.Jr=46;

% Pensions
Params.pension=1.3;

% Age-dependent labor productivity units
Params.kappa_j=[linspace(0.6,2,Params.Jr-15),linspace(2,1.5,14),zeros(1,Params.J-Params.Jr+1)];
% kappa_j increase from 0.6 to 2 from age j=1 to j=Jr-15, then decreases
% from 2 to 1.5 from age j=Jr-14 to j=Jr-1, and then is zero during retirement from j=Jr to j=J
% Note: Have made this unrealistically steep to make it easier to see the impact of borrowing constraints.

% Note: I call this kappa_j rather than kappa just to make it easier to remember that it 
% depend on j, it would make no difference to codes if it were just called kappa.


%% Grids
% Notice that we set minimum assets to -10, use 20 so maximum remains 10 (=20-10)
a_grid=-10+20*(linspace(0,1,n_a-1).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.
a_grid=sort([a_grid;0]); % Make sure there is a point specifically for zero (note, this code does not make sure there are not two zeros)
z_grid=1;
pi_z=1;

% No decision variable
d_grid=[];

%% Now, create the return function 
DiscountFactorParamNames={'beta'};

% Add r to the inputs (in some sense we add a and aprime, but these were already required, if previously irrelevant)
% Notice change to 'LifeCycleModel5_ReturnFn'
ReturnFn=@(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j,borrowingconstraint) LifeCycleModel15_ReturnFn(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j,borrowingconstraint)

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
jequaloneDist=zeros(n_a,1,'gpuArray'); % Put no households anywhere on grid
[~,aequalzeroindex]=min(abs(a_grid));
jequaloneDist(aequalzeroindex,1)=1; % Start with zero assets
% We have put all the 'new' households (mass of 1) here (zero assets)

%% We now compute the 'stationary distribution' of households
% This is effectively irrelevant to understanding life-cycle profiles but it is required as an input. 
% We will explain in a later model what the stationary distribution of households is and what we are doing here.
% We need to say how many agents are of each age (this is needed for the
% stationary distribution but is actually irrelevant to the life-cycle profiles)
Params.mewj=ones(1,Params.J)/Params.J; % Put a fraction 1/J at each age
AgeWeightsParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age
simoptions=struct(); % Use the default options
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
% Again, we will explain in a later model what the stationary distribution
% is, it is not important for our current goal of graphing the life-cycle profile

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (h,aprime,a,z) as first
% inputs, then just any relevant parameters.
FnsToEvaluate.earnings=@(aprime,a,z,w,kappa_j) w*kappa_j;
FnsToEvaluate.assets=@(aprime,a,z) a; % a is the current asset holdings
FnsToEvaluate.consumption=@(aprime,a,z,agej,Jr,w,kappa_j,r,pension) (agej<Jr)*(w*kappa_j+(1+r)*a-aprime)+(agej>=Jr)*(pension+(1+r)*a-aprime);
FnsToEvaluate.marginalutilityofcons=@(aprime,a,z,agej,Jr,w,kappa_j,r,pension,sigma) ((agej<Jr)*(w*kappa_j+(1+r)*a-aprime)+(agej>=Jr)*(pension+(1+r)*a-aprime))^(-sigma); % u(c)=(c^(1-sigma))/(1-sigma), therefore u'(c)=c^(-sigma)

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% For example
% AgeConditionalStats.earnings.Mean
% There are things other than Mean, but in our current deterministic model
% in which all agents are born identical the rest are meaningless.

%% Plot the life cycle profiles of fraction-of-time-worked, earnings, and assets

figure(1)
subplot(4,1,1); plot(1:1:Params.J,AgeConditionalStats.consumption.Mean)
ylim([0.5 2]) % Fix this so that can more easily see how it smooths when changing the borrowing constraint
title('Life Cycle Profile: Consumption (c)')
subplot(4,1,2); plot(1:1:Params.J,AgeConditionalStats.marginalutilityofcons.Mean)
title('Life Cycle Profile: Marginal Utility of Consumption (u''(c))')
ylim([0 3]) % Fix this so that can more easily see how drops when changing the borrowing constraint
subplot(4,1,3); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Labor Earnings (w kappa_j h)')
subplot(4,1,4); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Assets (a)')



