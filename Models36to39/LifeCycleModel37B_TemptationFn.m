function F=LifeCycleModel37B_TemptationFn(h,aprime,a,z,w,sigmatempt,scaletemptation,agej,Jr,pension,r,kappa_j)
% Assume the temptation is just in consumption (no temptation from
% leisure). Hence this is unchanged (except for taking h as first input)

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*z+(1+r)*a-aprime; % Add z here
else % Retirement
    c=pension+(1+r)*a-aprime;
end

if c>0
    F=scaletemptation*(c^(1-sigmatempt))/(1-sigmatempt); % The utility function
end


end
