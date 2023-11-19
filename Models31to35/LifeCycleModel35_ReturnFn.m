function F=LifeCycleModel35_ReturnFn(savings,riskyshare,hprime,h,a,z,w,sigma,agej,Jr,pension,kappa_j,sigma_h,f_htc,minhouse,rentprice,f_coll,houseservices)
% Note: riskyasset, so first inputs are (d,a,z,...)
% riskyshare is not used here, but still has to be input as it is part of d

% Make buying/selling a house costly/illiquid
htc=0; % house transaction cost
if hprime~=h
    htc=f_htc*(h+hprime);
end

% Housing services
if h==0
    s=0.5*houseservices*minhouse;
    rentalcosts=rentprice;
else
    s=houseservices*h;
    rentalcosts=0;
end

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*z+a-savings+(h-hprime)-htc-rentalcosts; % Note: +h and -hprime
else % Retirement
    c=pension+a-savings+(h-hprime)-htc-rentalcosts; % give a rent subsidy to elderly for no good reason-rentalcosts;
end

if c>0
    F=(((c^(1-sigma_h))*(s^sigma_h))^(1-sigma))/(1-sigma); % The utility function
end

if savings<-f_coll*hprime
    F=-Inf; % Collateral constraint on borrowing
end

if savings<0 && riskyshare>0
    F=-Inf; % Negative savings is only allowed in the form of a safe mortgage
end

%% Ban pensioners from negative assets
if agej>=Jr && savings<0
    F=-Inf;
end


end
