function [z_grid,pi_z]=LifeCycleModel21_ExogShockFn(agej,Jr)
% z takes two values:
% In working age: 1 is employment, 0 is unemployment
% In retirement: 0.3 is a medical expense, 0 is no expense
% (There is no particular point changing z_grid here, I just do it for clarity)

if agej<Jr
    z_grid=[1;0];
    pi_z=[0.7, 0.3; 0.5, 0.5]; % p_ee=0.7, p_eu=0.3, p_ue=0.5, p_uu=0.5
elseif agej==Jr % Want a special one-off transition to determine an initial distribution of the 'new' medical expense shocks
    z_grid=[0.3;0];
    pi_z=[0,1; 0,1]; % Everyone starts healthy (zero medical expense shock)
else
    z_grid=[0.3;0];
    pi_z=[0.2,0.8;0.3,0.7]; % Medical expense shocks are resonably rare and not very perisitent
end


end