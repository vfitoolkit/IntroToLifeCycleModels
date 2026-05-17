function housingservices=LifeCycleModel40_HousingServicesFn(aprime,hprime,a,h,z,w,r,p,theta,upsilon,phi,delta_o,agej,Jr,pension,kappa_j)
% Returns housing services consumed this period:
%   d for renters (chosen at price p; here solved analytically), hprime for owners.

% Income (working age vs retired)
if agej<Jr
    income=w*kappa_j*z;
else
    income=pension;
end

% Housing transactions cost (paid when h changes)
if hprime==h
    tau_hhprime=0;
else
    tau_hhprime=phi*h;
end

resources=income+(1+r)*a+(1-delta_o)*h-tau_hhprime;

if hprime==0
    %% Renter
    cspend=resources-aprime; % cspend=c+p*d
    if upsilon==0
        c=theta*cspend;
        d=(1-theta)*cspend/p;
    else
        c=cspend/(1+(p^(upsilon/(upsilon-1)))*((theta/(1-theta))^(1/(upsilon-1))));
        d=(cspend-c)/p;
    end
    housingservices=d;
else
    %% Owner
    housingservices=hprime;
end

end
