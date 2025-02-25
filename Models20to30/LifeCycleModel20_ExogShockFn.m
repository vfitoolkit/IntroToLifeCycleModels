function [z_grid,pi_z]=LifeCycleModel20_ExogShockFn(agej,Jr)
% z takes two values: 1 is employment, 0 is unemployment

z_grid=[1;0];

% Emprically unemployment rates are higher for the young, so we will change
% the transition matrix with age so that the duration of employment
% increases with age.
% One way to do this is to make the transition matrix at any age a linear
% combination of a 'young' matrix and an 'old' matrix, with the weights
% moving from young to old as agej increases.

pi_z_young=[0.5,0.5; 0.5,0.5]; % Probability of remaining employed is 0.5
pi_z_old=[0.9,0.1;0.5,0.5]; % Probability of remaining employed is 0.9

% I want weight on young to be one when agej=1
% I want weignt on old to be one when agej=Jr
% (Note: z is irrelavant to model when retired)

w=(Jr-agej)/(Jr-1); % =1 when agej=1, =0 when agej=Jr

if agej<=Jr
    pi_z=w*pi_z_young + (1-w)*pi_z_old;
else
    pi_z=[1,0;1,0]; % Just something silly, is not used for anything; just because above formula only works for agej<=Jr
end

end