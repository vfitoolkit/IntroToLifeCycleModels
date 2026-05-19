function consumption=LifeCycleModel40_ConsumptionFn(aprime,hprime,a,h,z,w,r,p,theta,upsilon,phi,delta_o,agej,Jr,pension,kappa_j)
% Returns nondurable consumption c this period.

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
    %% Renter: split cspend into c and p*d analytically (see
    % http://discourse.vfitoolkit.com/t/models-with-housing-a-simplification-to-handle-renters-splitting-cspend-analytically-into-consumption-c-and-housing-services-d/653)
    cspend=resources-aprime; % cspend=c+p*d
    if upsilon==0
        consumption=theta*cspend;
    else
        consumption=cspend/(1+(p^(upsilon/(upsilon-1)))*((theta/(1-theta))^(1/(upsilon-1))));
    end
else
    %% Owner: housing services come from hprime, so consumption is the residual
    consumption=resources-aprime-hprime;
end

end
