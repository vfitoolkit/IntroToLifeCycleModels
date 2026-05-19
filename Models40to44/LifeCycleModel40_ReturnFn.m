function F=LifeCycleModel40_ReturnFn(aprime,hprime,a,h,z,w,r,p,sigma,theta,upsilon,gamma,phi,delta_o,agej,Jr,pension,kappa_j,wg1,wg2,wg3,beta,sj)
% Two next-period endogenous states (aprime,hprime), two current endogenous
% states (a,h), one exogenous state (z), then parameters.

F=-Inf;

% Income (working age vs retired)
if agej<Jr
    income=w*kappa_j*z;
else
    income=pension;
end

% Housing transactions cost (paid when h changes; proportional to current house h)
if hprime==h
    tau_hhprime=0;
else
    tau_hhprime=phi*h;
end

% Resources available before purchasing new house and choosing aprime
resources=income+(1+r)*a+(1-delta_o)*h-tau_hhprime;

if hprime==0
    %% Renter: chooses housing services d at price p (solved analytically below)
    cspend=resources-aprime; % cspend=c+p*d
    if cspend>0
        % Analytic split of cspend into c and p*d from the CES first-order
        % condition with budget c+p*d=cspend. See
        % http://discourse.vfitoolkit.com/t/models-with-housing-a-simplification-to-handle-renters-splitting-cspend-analytically-into-consumption-c-and-housing-services-d/653
        % for the derivation.
        if upsilon==0
            % Cobb-Douglas limit
            c=theta*cspend;
            d=(1-theta)*cspend/p;
        else
            c=cspend/(1+(p^(upsilon/(upsilon-1)))*((theta/(1-theta))^(1/(upsilon-1))));
            d=(cspend-c)/p;
        end
        if c>0 && d>0
            if upsilon==0
                uinner=(c^theta)*(d^(1-theta));
            else
                uinner=(theta*(c^upsilon)+(1-theta)*(d^upsilon))^(1/upsilon);
            end
            F=(uinner^(1-sigma))/(1-sigma);
        end
    end
    % Renter cannot borrow
    if aprime<0
        F=-Inf;
    end
else
    %% Owner: housing services equal to the size of the house, hprime
    c=resources-aprime-hprime;
    if c>0
        if upsilon==0
            uinner=(c^theta)*(hprime^(1-theta));
        else
            uinner=(theta*(c^upsilon)+(1-theta)*(hprime^upsilon))^(1/upsilon);
        end
        F=(uinner^(1-sigma))/(1-sigma);
    end
    % Collateral constraint: mortgage cannot exceed (1-gamma) of the house value
    if aprime<-(1-gamma)*hprime
        F=-Inf;
    end
end

% Warm glow of bequest (only near end of life), on terminal financial+housing wealth
if agej>=Jr+10
    bequest=aprime+(1-delta_o)*hprime;
    if bequest>-wg2 % keep argument of (1+bequest/wg2) positive
        warmglow=wg1*((1+bequest/wg2)^(1-wg3))/(1-wg3);
        warmglow=beta*(1-sj)*warmglow; % only get warm glow next period if die
        F=F+warmglow;
    end
end

end
