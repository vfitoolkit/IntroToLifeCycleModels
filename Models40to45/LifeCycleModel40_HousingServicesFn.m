function housingservices=LifeCycleModel40_HousingServicesFn(aprime,hprime,a,h,z,kappaj,r,tau_p,theta,upsilon,phi,alpha,delta_k,delta_o,delta_r,agej,Jr,b)

housingservices=-Inf;

% price of housing
p=(r+delta_r)/(1+r);
% wage
w=(1-alpha)*((r+delta_k)/alpha)^(alpha/(alpha-1));

% housing transactions costs
if hprime==h
    tau_hhprime=0;
else
    tau_hhprime=phi*h;
end

% earnings
earnings=w*kappaj*z;

%% Renter
if hprime==0
    % Budget constraint
    cspend=(1+r)*a+(1-tau_p)*earnings+(1-delta_o)*h-tau_hhprime+(agej>=Jr)*b-aprime-hprime; % Note: -hprime=0
    % cspend=c+p*d (consumption goods plus housing services)
    % Analytically, we can derive the split of cspend into c and p*d as
    c=cspend/(1+(p^(upsilon/(upsilon-1)))*((theta/(1-theta))^(1/(upsilon-1))));
    d=(cspend-c)/p;

    housingservices=d;
%% Owner
elseif hprime>0
    housingservices=hprime;
end
