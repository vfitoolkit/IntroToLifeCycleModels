%% Life-Cycle Model 3: Assets
% We introduce assets, a consumption-savings decision.
% Assets is an endogenous state.
% Notice how the size of V and Policy change, it is worth trying to understand what these represent.

%% How does VFI Toolkit think about this?
%
% One decision variable: h, labour hours worked
% One endogenous state variable: a, assets (total household savings)
% No stochastic exogenous state variables
% Age: j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=51; % Endogenous labour choice (fraction of time worked)
n_a=201; % Endogenous asset holdings
n_z=0; % This is how the VFI Toolkit thinks about deterministic models
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

%% Grids
% The ^3 means that there are more points near 0 and near 10. We know from
% theory that the value function will be more 'curved' near zero assets,
% and putting more points near curvature (where the derivative changes the most) increases accuracy of results.
a_grid=10*(linspace(0,1,n_a).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.

% Grid for labour choice
h_grid=linspace(0,1,n_d)'; % Notice that it is imposing the 0<=h<=1 condition implicitly
% Switch into toolkit notation
d_grid=h_grid;

%% Now, create the return function 
DiscountFactorParamNames={'beta'};

% Add r to the inputs (in some sense we add a and aprime, but these were already required, if previously irrelevant)
% Notice change to 'LifeCycleModel3_ReturnFn'
ReturnFn=@(h,aprime,a,w,sigma,psi,eta,agej,Jr,pension,r) LifeCycleModel3_ReturnFn(h,aprime,a,w,sigma,psi,eta,agej,Jr,pension,r)

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
vfoptions=struct(); % Just using the defaults.
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, [], [], ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

% V is now (a,j). 
% Compare
size(V)
% with
[n_a,N_j]
% there are the same.
% Policy is
size(Policy)
% which is the same as
[length(n_d)+length(n_a),n_a,N_j]
% The n_a,n_z,N_j represent the state on which the decisions/policys
% depend, and there is one decision for each decision variable 'd' and each endogenous state variable 'a'

%% Let's take a quick look at what we have calculated, namely V and Policy

% The value function V depends on the state, so now it depends on both asset holdings and age.

% We can plot V as a 3d plot (surf is matlab command for 3d plot)
figure(1)
surf(a_grid*ones(1,Params.J),ones(n_a,1)*(1:1:Params.J),V) % The reshape is needed to get rid of z
title('Value function')
xlabel('Assets (a)')
ylabel('Age j')

% Do another plot of V, this time as a function (of assets) for a given age (I do a few for different ages)
figure(2)
subplot(5,1,1); plot(a_grid,V(:,1)) % j=1
title('Value fn at age j=1')
subplot(5,1,2); plot(a_grid,V(:,20)) % j=20
title('Value fn at age j=20')
subplot(5,1,3); plot(a_grid,V(:,45)) % j=45
title('Value fn at age j=45')
subplot(5,1,4); plot(a_grid,V(:,46)) % j=46
title('Value fn at age j=46 (first year of retirement)')
subplot(5,1,5); plot(a_grid,V(:,81)) % j=81
title('Value fn at age j=81')
xlabel('Assets (a)')

% Convert the policy function to values (rather than indexes).
% Note that there is one policy for hours worked (h), and another for next
% period assets (aprime). 
% Policy(1,:,:) is h, Policy(2,:,:) is aprime [as function of (a,j)]
% Plot both as a 3d plot.
figure(3)
PolicyVals=PolicyInd2Val_FHorz_Case1(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid);
subplot(2,1,1); surf(a_grid*ones(1,Params.J),ones(n_a,1)*(1:1:Params.J),reshape(PolicyVals(1,:,:),[n_a,Params.J]))
title('Policy function: fraction of time worked (h)')
xlabel('Assets (a)')
ylabel('Age j')
zlabel('Fraction of Time Worked (h)')
subplot(2,1,2); surf(a_grid*ones(1,Params.J),ones(n_a,1)*(1:1:Params.J),reshape(PolicyVals(2,:,:),[n_a,Params.J]))
title('Policy function: next period assets (aprime)')
xlabel('Assets (a)')
ylabel('Age j')
zlabel('Next period assets (aprime)')

% Again, plot both policies (h and aprime), this time as a function (of assets) for a given age  (I do a few for different ages)
figure(4)
subplot(5,2,1); plot(a_grid,PolicyVals(1,:,1)) % j=1
title('Policy for h at age j=1')
subplot(5,2,3); plot(a_grid,PolicyVals(1,:,20)) % j=20
title('Policy for h at age j=20')
subplot(5,2,5); plot(a_grid,PolicyVals(1,:,45)) % j=45
title('Policy for h at age j=45')
subplot(5,2,7); plot(a_grid,PolicyVals(1,:,46)) % j=46
title('Policy for h at age j=46 (first year of retirement)')
subplot(5,2,9); plot(a_grid,PolicyVals(1,:,81)) % j=81
title('Policy for h at age j=81')
xlabel('Assets (a)')
subplot(5,2,2); plot(a_grid,PolicyVals(2,:,1)) % j=1
title('Policy for aprime at age j=1')
subplot(5,2,4); plot(a_grid,PolicyVals(2,:,20)) % j=20
title('Policy for aprime at age j=20')
subplot(5,2,6); plot(a_grid,PolicyVals(2,:,45)) % j=45
title('Policy for aprime at age j=45')
subplot(5,2,8); plot(a_grid,PolicyVals(2,:,46)) % j=46
title('Policy for aprime at age j=46 (first year of retirement)')
subplot(5,2,10); plot(a_grid,PolicyVals(2,:,81)) % j=81
title('Policy for aprime at age j=81')
xlabel('Assets (a)')

