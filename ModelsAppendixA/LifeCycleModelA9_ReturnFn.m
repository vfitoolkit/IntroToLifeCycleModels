function F=LifeCycleModelA9_ReturnFn(h,f,search,aprime,a,n,work,z,w,sigma,psi,eta,agej,eta1,eta2,eta3,nbar,childcarec,Jr,pension,r,kappa_j,benefits,search_c)

F=-Inf;

childcarecosts=childcarec*n*(h>0); % Cost of childcare if working

c=-1; % gpu objected that c might not be initiated, so adding this line
if agej<Jr % If working age
    if work==1
        c=w*kappa_j*z*h+(1+r)*a-childcarecosts-aprime;
    elseif work==0
        c=benefits+(1+r)*a-childcarecosts-aprime;
    end
else % Retirement
    c=pension+(1+r)*a-aprime;
end

utility_of_children=eta1*exp(agej-eta3)/(1+exp(agej-eta3)) *(nbar+n)^eta2;
dislike_search=search_c*search;

if c>0
    F=(c^(1-sigma))/(1-sigma)-psi*(h^(1+eta2))/(1+eta) +utility_of_children-dislike_search;
    % Utility is: +consumption, -hours worked, +children, -search effort
end



end
