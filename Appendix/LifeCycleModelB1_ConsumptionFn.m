function c=LifeCycleModelB1_ConsumptionFn(h,aprime,a,z,w,agej,Jr,pension,r,kappa_j)
% Note: this was created by just editing the return function so that it just returns consumption instead

if agej<Jr % If working age
    c=w*kappa_j*z*h+(1+r)*a-aprime; % Add z here
else % Retirement
    c=pension+(1+r)*a-aprime;
end

end
