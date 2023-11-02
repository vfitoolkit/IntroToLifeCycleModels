function aprime=LifeCycleModel31_aprimeFn(savings,riskyshare,u, r)
% Note: because of how riskyasset works we need to input (d1,d2,u) as the first arguements.
% That is, the first inputs must be the decision variables (d variables),
% followed by the shocks that are iid and occur between periods (u variables)

aprime=(1+r)*(1-riskyshare)*savings+(1+r+u)*riskyshare*savings;

end