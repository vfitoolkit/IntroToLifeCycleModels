function [V, Policy]=ValueFnIter_Case1_FHorz_nod_raw(n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%
a_grid=gpuArray(a_grid);

if vfoptions.lowmemory>0
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
    % z_gridvals is created below
end
if vfoptions.lowmemory>1
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1); % The 1 at end indicates want output in form of matrix.
end


%% j=N_j
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
        end

    elseif vfoptions.lowmemory==2

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for a_c=1:N_a
                a_val=a_gridvals(a_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec,0);
                % Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_az);
                V(a_c,z_c,N_j)=Vtemp;
                Policy(a_c,z_c,N_j)=maxindex;

            end
        end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    if vfoptions.lowmemory==0

        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);

        if vfoptions.paroverz==1
            % (aprime,a,z)

            % Use sparse for a few lines until sum over zprime
            EV=V_Jplus1.*shiftdim(pi_z_J(:,:,N_j)',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV.*ones(1,N_a,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V(:,:,N_j)=shiftdim(Vtemp,1);
            Policy(:,:,N_j)=shiftdim(maxindex,1);

        elseif vfoptions.paroverz==0

            for z_c=1:N_z
                ReturnMatrix_z=ReturnMatrix(:,:,z_c);

                %Calc the condl expectation term (except beta), which depends on z but
                %not on control variables
                EV_z=V_Jplus1.*pi_z_J(z_c,:,N_j);
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);

                entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V(:,z_c,N_j)=Vtemp;
                Policy(:,z_c,N_j)=maxindex;
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);

            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=V_Jplus1.*pi_z_J(z_c,:,N_j);
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);

            entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=V_Jplus1.*pi_z_J(z_c,:,N_j);
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);

            z_val=z_gridvals_J(z_c,:,N_j);
            for a_c=1:N_a
                a_val=a_gridvals(a_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec,0);

                entireRHS_az=ReturnMatrix_az+DiscountFactorParamsVec*EV_z;
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_az);
                V(a_c,z_c,N_j)=Vtemp;
                Policy(a_c,z_c,N_j)=maxindex;
            end
        end

    end
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    VKronNext_j=V(:,:,jj+1);
    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,0);
        
        if vfoptions.paroverz==1
            % (aprime,a,z)
            
            % Use sparse for a few lines until sum over zprime
            EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension
            
            entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV.*ones(1,N_a,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);
            
            V(:,:,jj)=shiftdim(Vtemp,1);
            Policy(:,:,jj)=shiftdim(maxindex,1);
            
        elseif vfoptions.paroverz==0
            
            for z_c=1:N_z
                ReturnMatrix_z=ReturnMatrix(:,:,z_c);
                
                %Calc the condl expectation term (except beta), which depends on z but
                %not on control variables
                EV_z=VKronNext_j.*pi_z_J(z_c,:,jj);
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);
                
                entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V(:,z_c,jj)=Vtemp;
                Policy(:,z_c,jj)=maxindex;
            end
        end
        
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);
            
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=VKronNext_j.*pi_z_J(z_c,:,jj);
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,jj)=Vtemp;
            Policy(:,z_c,jj)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=VKronNext_j.*pi_z_J(z_c,:,jj);
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
                        
            z_val=z_gridvals_J(z_c,:,jj);
            for a_c=1:N_a
                a_val=a_gridvals(a_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec,0);
                
                entireRHS_az=ReturnMatrix_az+DiscountFactorParamsVec*EV_z;
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_az);
                V(a_c,z_c,jj)=Vtemp;
                Policy(a_c,z_c,jj)=maxindex;
            end
        end
        
    end
end


end