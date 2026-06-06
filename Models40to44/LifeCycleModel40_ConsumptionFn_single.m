function consumption=LifeCycleModel40_ConsumptionFn_single(aprime,hprime,a,h,z,w,r,p,theta,upsilon,phi,delta_o,agej,Jr,pension,kappa_j)
% Returns nondurable consumption c this period.

single_1=single(1);

% Income (working age vs retired)
if agej<Jr
    income=w*kappa_j*z;
else
    income=pension;
end

% Housing transactions cost (paid when h changes)
if hprime==h
    tau_hhprime=single(0);
else
    tau_hhprime=phi*h;
end

resources=income+(single_1+r)*a+(single_1-delta_o)*h-tau_hhprime;

if hprime==0
    %% Renter: split cspend into c and p*d analytically
    cspend=resources-aprime; % cspend=c+p*d
    if upsilon==0
        consumption=theta*cspend;
    else
        consumption=cspend/(single_1+(p^(upsilon/(upsilon-single_1)))*((theta/(single_1-theta))^(single_1/(upsilon-single_1))));
    end
else
    %% Owner: housing services come from hprime, so consumption is the residual
    consumption=resources-aprime-hprime;
end

end
