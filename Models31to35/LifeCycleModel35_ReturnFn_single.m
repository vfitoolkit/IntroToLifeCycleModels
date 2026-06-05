function F=LifeCycleModel35_ReturnFn_single(savings,hprime,h,a,z,w,sigma,agej,Jr,pension,kappa_j,sigma_h,f_htc,minhouse,rentprice,f_coll,houseservices)
% Note: riskyasset, so first inputs are (d,a,z,...)
% vfoptions.refine_d: only decisions d1,d3 are input to ReturnFn (and this model has no d1)

% Make buying/selling a house costly/illiquid
single_0=single(0);
htc=single_0; % house transaction cost
if hprime~=h
    htc=f_htc*(h+hprime);
end

% Housing services
if h==0
    s=single(0.5)*houseservices*minhouse;
    rentalcosts=rentprice;
else
    s=houseservices*h;
    rentalcosts=single_0;
end

F=single(-Inf);
if agej<Jr % If working age
    c=w*kappa_j*z+a-savings+(h-hprime)-htc-rentalcosts; % Note: +h and -hprime
else % Retirement
    c=pension+a-savings+(h-hprime)-htc-rentalcosts; % give a rent subsidy to elderly for no good reason-rentalcosts;
end

if c>0
    single_1=single(1);
    F=(((c^(single_1-sigma_h))*(s^sigma_h))^(single_1-sigma))/(single_1-sigma); % The utility function
end

if savings<-f_coll*hprime
    F=single(-Inf); % Collateral constraint on borrowing
end

% Negative savings is only allowed in the form of a safe mortgage. This is dealt with via the aprimeFn.

%% Ban pensioners from negative assets
if agej>=Jr && savings<0
    F=single(-Inf);
end


end
