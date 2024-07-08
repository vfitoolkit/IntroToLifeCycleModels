function [V,Policy]=ValueFnIter_Case1_FHorz_SemiExo_DC1(n_d1,n_d2, n_a, n_z, n_semiz, N_j, d1_grid, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_z=prod(n_z);


if length(n_a)==1
    if N_d1==0
        if isfield(vfoptions,'n_e')
            error('Have not yet implemented divideandconquer for semi-exo with an e variable (contact me)')
            % if N_z==0
                % [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC1_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            % else
                % [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_DC1_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            % end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_SemiExo_DC1_nod1_noz_raw(n_d2,n_a,n_semiz, N_j, d2_grid, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_SemiExo_DC1_nod1_raw(n_d2,n_a,n_z,n_semiz, N_j, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else % N_d
        if isfield(vfoptions,'n_e')
            error('Have not yet implemented divideandconquer for semi-exo with an e variable (contact me)')
            if N_z==0
                % [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC1_noz_e_raw(n_d,n_a,  vfoptions.n_e, N_j, d_grid, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                % [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC1_e_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_grid, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_SemiExo_DC1_noz_raw(n_d1, n_d2,n_a,n_semiz, N_j, d1_grid, d2_grid, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_SemiExo_DC1_raw(n_d1, n_d2,n_a,n_z,n_semiz, N_j, d1_grid, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
else
    error('vfoptions.divideandconquer==1 is currently only possible for one endogenous state (so length(n_a)=1)')
end


%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==0
    if n_d1==0
        n_d=n_d2;
    else
        n_d=[n_d1,n_d2];
    end
    if isfield(vfoptions,'n_e')
        if N_z==0
            V=reshape(VKron,[n_a,n_semiz,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, [n_semiz,vfoptions.n_e], N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_semiz,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz_e(PolicyKron, n_d, n_a, [n_semiz,n_z], vfoptions.n_e, N_j, vfoptions);
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,n_semiz,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a,n_semiz, N_j, vfoptions);
        else
            V=reshape(VKron,[n_a,n_semiz,n_z,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, [n_semiz,n_z], N_j, vfoptions);
        end
    end
else
    V=VKron;
    Policy=PolicyKron;
end

% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1
    fprintf('USING vfoptions to force integer... \n')
    % First, give some output on the size of any changes in Policy as a
    % result of turning the values into integers
    temp=max(max(max(abs(round(Policy)-Policy))));
    while ndims(temp)>1
        temp=max(temp);
    end
    fprintf('  CHECK: Maximum change when rounding values of Policy is %8.6f (if these are not of numerical rounding error size then something is going wrong) \n', temp)
    % Do the actual rounding to integers
    Policy=round(Policy);
    % Somewhat unrelated, but also do a double-check that Policy is now all positive integers
    temp=min(min(min(Policy)));
    while ndims(temp)>1
        temp=min(temp);
    end
    fprintf('  CHECK: Minimum value of Policy is %8.6f (if this is <=0 then something is wrong) \n', temp)
%     Policy=uint64(Policy);
%     Policy=double(Policy);
elseif vfoptions.policy_forceintegertype==2
    % Do the actual rounding to integers
    Policy=round(Policy);
end




end
