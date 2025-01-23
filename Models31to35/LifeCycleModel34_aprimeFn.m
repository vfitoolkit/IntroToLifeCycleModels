function aprime=LifeCycleModel34_aprimeFn(riskyshare,savings,u, r)
% Note: because of how riskyasset works we need to input (d,u,...) as the first arguements.
% That is, the first inputs must be the decision variables (d variables),
% followed by the shocks that are iid and occur between periods (u variables)
% And because we use vfoptions.refine_d, the decision variables for aprimeFn must follow the ordering d2,d3

if savings>0
    aprime=(1+r)*(1-riskyshare)*savings+(1+r+u)*riskyshare*savings;
else
    % following is enforcing the idea that negative savings represents a mortgage, and so must involve riskyshare=0
    aprime=(1+r)*savings;
end

end