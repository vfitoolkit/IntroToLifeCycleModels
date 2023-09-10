function c=LifeCycleModel38_ConsumptionFn(aprime,a,clag,z,w,agej,Jr,pension,r,kappa_j)

if agej<Jr % If working age
    c=w*kappa_j*z+(1+r)*a-aprime; % Add z here
else % Retirement
    c=pension+(1+r)*a-aprime;
end

end
