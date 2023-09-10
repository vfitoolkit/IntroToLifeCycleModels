function F=LifeCycleModel36_ReturnFn(aprime,a,z,w,sigma,agej,Jr,pension,r,kappa_j)

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*z+(1+r)*a-aprime; % Add z here
else % Retirement
    c=pension+(1+r)*a-aprime;
end

if c>0
    F=(c^(1-sigma))/(1-sigma); % The utility function
end


end
