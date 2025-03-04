function F=LifeCycleModel27_ReturnFn(aprime,a,z1,upsilon,e,alpha,w,sigma,agej,Jr,pension,incomefloor,r,kappa_j,wg1,wg2,wg3,beta,sj,tau1,tau2,Jbeq)
% Exogenous labor-supply model
% Earning dynamics process following GKOS2021

F=-Inf;
if agej<Jr % If working age
    Income=w*(1-upsilon)*exp(kappa_j+alpha+z1+e)+r*a;
    if Income>0
        IncomeTax=tau1+tau2*log(Income)*Income;
    else
        IncomeTax=0;
    end
    AfterTaxIncome=Income-IncomeTax;
    if AfterTaxIncome<incomefloor
        AfterTaxIncome=incomefloor;
    end
    c=AfterTaxIncome+a-aprime;
else % Retirement
    Income=r*a;
    if Income>0
        IncomeTax=tau1+tau2*log(Income)*Income;
    else
        IncomeTax=0;
    end
    % Income floor is not relevant as all get pension (and pension>incomefloor)
    c=pension+(Income-IncomeTax)+a-aprime;
end

if c>0
    F=(c^(1-sigma))/(1-sigma); % The utility function
end

% add the warm glow to the return, but only near end of life
if agej>=Jbeq
    % Warm glow of bequests (use functional form of De Nardi (2004))
    warmglow=wg1*((1+aprime/wg2)^(1-wg3))/(1-wg3);
    % Modify for beta and sj (get the warm glow next period if die)
    warmglow=beta*(1-sj)*warmglow;
    % add the warm glow to the return
    if c>0 % I don't think this should be needed, but added to be sure
        F=F+warmglow;
    end
end

end
