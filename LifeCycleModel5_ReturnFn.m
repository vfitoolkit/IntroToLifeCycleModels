function F=LifeCycleModel5_ReturnFn(h,aprime,a,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j)
% Notice that we just use kappa_j directly. VFI Toolkit handles
% age-dependent parameters by noticing they depend on age (because they are
% of length N_j) and then chooses the value for the appropriate age to pass
% to the ReturnFn. Hence inside the ReturnFn we can just use kappa_j
% directly, and we know that the toolkit will make sure that the value for
% the relevant age is being used.

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*h+(1+r)*a-aprime; % add assets, and subtract 'savings'
else % Retirement
    c=pension+(1+r)*a-aprime; % add assets, and subtract 'savings'
end

if c>0
    F=(c^(1-sigma))/(1-sigma) -psi*(h^(1+eta))/(1+eta); % The utility function
end

end
