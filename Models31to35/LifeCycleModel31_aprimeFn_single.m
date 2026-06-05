function aprime=LifeCycleModel31_aprimeFn_single(riskyshare,savings,u, r)
% Note: because of how riskyasset works we need to input (d,u,...) as the first arguements.
% That is, the first inputs must be the decision variables (d variables),
% followed by the shocks that are iid and occur between periods (u variables)
% And because we use vfoptions.refine_d, the decision variables for aprimeFn must follow the ordering d2,d3

single_1=single(1);
aprime=(single_1+r)*(single_1-riskyshare)*savings+(single_1+r+u)*riskyshare*savings;

end