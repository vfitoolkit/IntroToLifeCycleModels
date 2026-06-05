%% Life-Cycle Model 1: Consumption-Leisure
% Agent lives for J periods
% Each period the only decision is 'hours worked' h
% This model is so simple as to be slightly silly, but you have to start somewhere ;)

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
n_a=1; % Codes require an endogenous state, but by making it only one grid point it is essentially irrelevant
n_z=0; % This is how the VFI Toolkit thinks about deterministic models
N_j=Params.J; % Number of periods in finite horizon

%% Parameters

% Discount rate
Params.beta = 0.96;
% Preferences
Params.sigma = 2; % Coeff of relative risk aversion (curvature of consumption)
Params.eta = 1.5; % Curvature of leisure (This will end up being 1/Frisch elasticity)
Params.psi = 10; % Weight on leisure

Params.w=1; % Wage

%% Grids
precision='single';
precision_cast=@(x) single(x)

% While there are no 'a' for 'z' in this model, VFI Toolkit requires them 
% to figure out what is going on. By making them just a single grid point, 
% and then not using them anywhere, we are essentially solving a model without them.
a_grid=precision_cast(1);

% Grid for labour choice
h_grid=linspace(precision_cast(0),precision_cast(1),n_d)'; % Notice that it is imposing the 0<=h<=1 condition implicitly
% Switch into toolkit notation
d_grid=h_grid;

%% Now, create the return function

% We tell VFI Toolkit the name of which parameter is the discount factor
DiscountFactorParamNames={'beta'};

% When setting up the return function the first 'inputs' must be the
% decision variables, next period endogenous states, this period endogenous states.
% In the current model, the decision variable is h.
% We don't have an endogenous state, but toolkit requires one and this is
% the 'a', which we created as a single grid point. We will call the next
% period version of this aprime. We don't have an exogenous state so we set
% n_z=0.
% So 'decision variables, next period endogenous states, this period
% endogenous states' becomes
% (h,aprime,a)
% After this we can put the parameters.

% To understand how to create the ReturnFn, look inside
% 'LifeCycleModel1_ReturnFn' (you can right-click on its name below and click 'Open LifeCycleModel1_ReturnFn')
% We then just have to make the @() contain exactly the same inputs as
% 'LifeCycleModel1_ReturnFn', and then give the parameter names.
if strcmp(precision,'single')
    ReturnFn=@(h,aprime,a,w,sigma,psi,eta)...
        LifeCycleModel1_ReturnFn_single(h,aprime,a,w,sigma,psi,eta);
else
    ReturnFn=@(h,aprime,a,w,sigma,psi,eta)...
        LifeCycleModel1_ReturnFn(h,aprime,a,w,sigma,psi,eta);
end

% The first entries must be the decision variables (d), the next period endogenous 
% state (aprime), and this period endogenous state (a), followed by any parameters.
% VFI Toolkit will automatically look in 'Params' to find the values of these parameters.

% A standard endogenous state in VFI Toolkit is one where you can choose
% next period endogenous state directly. Hence why we have both of (..,aprime,a,..)
% in the ReturnFn.

%% Solve the value function iteration problem
disp('Solve for Value fn and Policy fn using ValueFnIter command')
vfoptions=struct(); % Just using the defaults.
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, [], [], ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

% V is the value function
% Policy is the policy function (as index)
% Policy(1,:,:) is the decision variable
% Policy(2,:,:) is next period asset (which is trivial in this model)

%% Let's take a quick look at what we have calculated, namely V and Policy

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
% size(V) is 1-by-1-by-N_j (the first two dimensions are the endogenous
% state and exogenous state, but in this model they are trivial)
% We use shiftdim(V,1) which will move all the dimension left once,
% so we get a 1-by-J result that we can then plot (you cannot plot three
% dimensional arrays)

% Notice that the value function takes negative values. The value function
% is the 'present discounted utility'. Just like utility is only defined up
% to a linear transformation so is the value function. So we could add a
% constant to the utility function (or multiply it by a constant) without
% changing the preferences it represents. As a result it is not important
% that utility is negative (we could just add a big constant and make it
% positive) and the same follows for the value function.

% Plot the policy function (the values of h chosen at each age).
% Policy by default stores grid-point *indexes*, not the values themselves,
% so we use the toolkit helper PolicyInd2Val_FHorz() to convert them
% into the actual h values. This is what you will use in practice -- it
% generalises to more decision variables, multiple endogenous states, etc.
figure(2)
PolicyVals=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions);
plot(1:1:Params.J,shiftdim(PolicyVals(1,:,:),2))
title('Policy function: fraction of time worked (h)')
xlabel('Age j')
% Looks a bit silly, people of every age work exactly the same amount. This will change in future models.

% To see what PolicyInd2Val_FHorz() is doing under the hood, we can
% do the index-to-value conversion by hand: just index into h_grid using
% Policy. This gives the same plot, but only works because this model is
% simple and so the contents of Policy are simple; doing this manually
% becomes tricky in more advanced models and it is recommended you always
% use PolicyInd2Val_FHorz().
figure(3)
plot(1:1:Params.J,h_grid(Policy(1,:,:))) % h_grid(Policy) turns grid point index into actual values of h
title('Policy function: fraction of time worked (h)')
xlabel('Age j')




