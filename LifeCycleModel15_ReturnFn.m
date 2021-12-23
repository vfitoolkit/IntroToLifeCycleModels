function F=LifeCycleModel15_ReturnFn(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j,borrowingconstraint)

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j+(1+r)*a-aprime; % add assets, and subtract 'savings'
else % Retirement
    c=pension+(1+r)*a-aprime; % add assets, and subtract 'savings'
end

if c>0
    F=(c^(1-sigma))/(1-sigma); % The utility function
end

if aprime<borrowingconstraint
    F=-Inf; % You cannot chose assets less than the borrowing constraint 
end

end
