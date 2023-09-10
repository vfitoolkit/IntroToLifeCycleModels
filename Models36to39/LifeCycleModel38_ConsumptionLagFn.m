function c=LifeCycleModel38_ConsumptionLagFn(aprime,a,z,w,agej,Jr,pension,r,kappa_j)

if agej<Jr % If working age
    c=w*kappa_j*z+(1+r)*a-aprime; % Add z here
else % Retirement
    c=pension+(1+r)*a-aprime;
end

end
