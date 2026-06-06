function F=LifeCycleModel40_ReturnFn_single(aprime,hprime,a,h,z,w,r,p,sigma,theta,upsilon,gamma,phi,delta_o,agej,Jr,pension,kappa_j,wg1,wg2,wg3,beta,sj)
% Two next-period endogenous states (aprime,hprime), two current endogenous
% states (a,h), one exogenous state (z), then parameters.

F=single(-Inf);
single_1=single(1);

% Income (working age vs retired)
if agej<Jr
    income=w*kappa_j*z;
else
    income=pension;
end

% Housing transactions cost (paid when h changes; proportional to current house h)
if hprime==h
    tau_hhprime=single(0);
else
    tau_hhprime=phi*h;
end

% Resources available before purchasing new house and choosing aprime
resources=income+(single_1+r)*a+(single_1-delta_o)*h-tau_hhprime;

if hprime==0
    %% Renter: chooses housing services d at price p (solved analytically below)
    cspend=resources-aprime; % cspend=c+p*d
    if cspend>0
        % Analytic split of cspend into c and p*d from the CES first-order
        % condition with budget c+p*d=cspend.
        if upsilon==0
            % Cobb-Douglas limit
            c=theta*cspend;
            d=(single_1-theta)*cspend/p;
        else
            c=cspend/(single_1+(p^(upsilon/(upsilon-single_1)))*((theta/(single_1-theta))^(single_1/(upsilon-single_1))));
            d=(cspend-c)/p;
        end
        if c>0 && d>0
            if upsilon==0
                uinner=(c^theta)*(d^(single_1-theta));
            else
                uinner=(theta*(c^upsilon)+(single_1-theta)*(d^upsilon))^(single_1/upsilon);
            end
            F=(uinner^(single_1-sigma))/(single_1-sigma);
        end
    end
    % Renter cannot borrow
    if aprime<0
        F=single(-Inf);
    end
else
    %% Owner: housing services equal to the size of the house, hprime
    c=resources-aprime-hprime;
    if c>0
        if upsilon==0
            uinner=(c^theta)*(hprime^(single_1-theta));
        else
            uinner=(theta*(c^upsilon)+(single_1-theta)*(hprime^upsilon))^(single_1/upsilon);
        end
        F=(uinner^(single_1-sigma))/(single_1-sigma);
    end
    % Collateral constraint: mortgage cannot exceed (1-gamma) of the house value
    if aprime<-(single_1-gamma)*hprime
        F=single(-Inf);
    end
end

% Warm glow of bequest (only near end of life), on terminal financial+housing wealth
if agej>=Jr+10
    bequest=aprime+(single_1-delta_o)*hprime;
    if bequest>-wg2 % keep argument of (1+bequest/wg2) positive
        warmglow=wg1*((single_1+bequest/wg2)^(single_1-wg3))/(single_1-wg3);
        warmglow=beta*(single_1-sj)*warmglow; % only get warm glow next period if die
        F=F+warmglow;
    end
end

end
