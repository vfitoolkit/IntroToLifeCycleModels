%% Life-Cycle Model 2: Retirement
% In the previous model nothing really changes as agent gets older, let's have 'working age' and 'retirement'
% Agent lives for J periods. Retires in period Jr.
% Each period before retirement the only decision is 'hours worked' h
% In retirement, no working, just receive pension

%% How does VFI Toolkit think about this?
%
% One decision variable: h, labour hours worked
% No endogenous state variable
% No stochastic exogenous state variables
% Age: j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=51; % Endogenous labour choice (fraction of time worked)
n_a=1; % Codes require an endogeneous state, but by making it only one grid point it is essentially irrelevant
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

% Demographics
Params.agej=1:1:Params.J; % Is a vector of all the agej: 1,2,3,...,J
Params.Jr=46;

% Pensions
Params.pension=0.3;

%% Grids
% While there are no 'a' in this model, VFI Toolkit requires them 
% to figure out what is going on. By making it just a single grid point, 
% and then not using it anywhere, we are essentially solving a model without them.
a_grid=1;

% Grid for labour choice
h_grid=linspace(0,1,n_d)'; % Notice that it is imposing the 0<=h<=1 condition implicitly
% Switch into toolkit notation
d_grid=h_grid;

%% Now, create the return function 

DiscountFactorParamNames={'beta'};

% Add agej,Jr & pension to the inputs
% Notice change to 'LifeCycleModel2_ReturnFn'
ReturnFn=@(h,aprime,a,w,sigma,psi,eta,agej,Jr,pension) LifeCycleModel2_ReturnFn(h,aprime,a,w,sigma,psi,eta,agej,Jr,pension);
% VFI Toolkit will automatically look in 'Params' to find the values of these parameters.

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
vfoptions=struct(); % Just using the defaults.
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, [], [], ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

%% Let's take a quick look at what we have calculated, namely V and Policy

% Obviously now that we have added retirement noone works during retirement.

% Plot of Value function in terms of age j, and a second in terms of age in
% years (remember period j=1 represents age 20). You can think of the value
% fn as representing the utility at that age.
figure(1)
subplot(2,1,1); plot(1:1:Params.J,shiftdim(V,1))
title('Value function')
xlabel('Age j')
subplot(2,1,2); plot(Params.agejshifter+(1:1:Params.J),shiftdim(V,1))
title('Value function')
xlabel('Age in Years')

% Plot the policy function, which represents the (grid points relating to) values of h.
figure(2)
plot(1:1:Params.J,h_grid(Policy(1,:,:))) % Notice how it is h_grid(Policy), this turns grid point index into actual values of h
title('Policy function: fraction of time worked (h)')
xlabel('Age j')
% Notice how now with retirement no-one works (retirement is at age Param.Jr)

% There is actually also a command for converting Policy into policy values (rather than policy indexes, which is default)
figure(3)
PolicyVals=PolicyInd2Val_FHorz_Case1(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid);
plot(1:1:Params.J,shiftdim(PolicyVals(1,:,:),2))
title('Policy function: fraction of time worked (h)')
xlabel('Age j')





