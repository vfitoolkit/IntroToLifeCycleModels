function [V, Policy]=ValueFnIter_Case1_FHorz_GulPesendorfer_nod_raw(n_a,n_z,N_j, a_grid, z_grid,pi_z, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions)

N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%
a_grid=gpuArray(a_grid);
z_grid=gpuArray(z_grid);

eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')

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
TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames, N_j);

if fieldexists_pi_z_J==1
    z_grid=vfoptions.z_grid_J(:,N_j);
    pi_z=vfoptions.pi_z_J(:,:,N_j);
elseif fieldexists_ExogShockFn==1
    if fieldexists_ExogShockFnParamNames==1
        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,N_j);
        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
        for ii=1:length(ExogShockFnParamsVec)
            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
        end
        [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
        z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    else
        [z_grid,pi_z]=vfoptions.ExogShockFn(N_j);
        z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    end
end
if vfoptions.lowmemory>0
    if all(size(z_grid)==[sum(n_z),1])
        z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(z_grid)==[prod(n_z),l_z])
        z_gridvals=z_grid;
    end
end

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec,0);

        TemptationMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(TemptationFn, 0, n_a, n_z, 0, a_grid, z_grid, TemptationFnParamsVec,0);
        MostTempting=max(TemptationMatrix,[],1);
        entireRHS=ReturnMatrix+TemptationMatrix-ones(N_a,1).*MostTempting;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);

            TemptationMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(TemptationFn, 0, n_a, special_n_z, 0, a_grid, z_val, TemptationFnParamsVec,0);
            MostTempting_z=max(TemptationMatrix_z,[],1);
            entireRHS_z=ReturnMatrix_z+TemptationMatrix_z-ones(N_a,1).*MostTempting_z;

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
        end

    elseif vfoptions.lowmemory==2

        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            for a_c=1:N_a
                a_val=a_gridvals(a_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec,0);

                TemptationMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(TemptationFn, 0, special_n_a, special_n_z, 0, a_val, z_val, TemptationFnParamsVec,0);
                MostTempting_az=max(TemptationMatrix_az,[],1);
                entireRHS_az=ReturnMatrix_az+TemptationMatrix_az-ones(N_a,1).*MostTempting_az;

                % Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_az);
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

        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec,0);

        TemptationMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(TemptationFn, 0, n_a, n_z, 0, a_grid, z_grid, TemptationFnParamsVec,0);
        MostTempting=max(TemptationMatrix,[],1);

        if vfoptions.paroverz==1
            % (aprime,a,z)

            % Use sparse for a few lines until sum over zprime
            EV=V_Jplus1.*shiftdim(pi_z',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            entireRHS=ReturnMatrix+TemptationMatrix-ones(N_a,1).*MostTempting+DiscountFactorParamsVec*EV.*ones(1,N_a,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V(:,:,N_j)=shiftdim(Vtemp,1);
            Policy(:,:,N_j)=shiftdim(maxindex,1);

        elseif vfoptions.paroverz==0

            for z_c=1:N_z
                ReturnMatrix_z=ReturnMatrix(:,:,z_c);
                TemptationMatrix_z=ReturnMatrix(:,:,z_c);
                MostTempting_z=MostTempting(1,:,z_c);

                %Calc the condl expectation term (except beta), which depends on z but
                %not on control variables
                EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);

                entireRHS_z=ReturnMatrix_z+TemptationMatrix_z-ones(N_a,1).*MostTempting_z+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V(:,z_c,N_j)=Vtemp;
                Policy(:,z_c,N_j)=maxindex;
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);

            TemptationMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(TemptationFn, 0, n_a, special_n_z, 0, a_grid, z_val, TemptationFnParamsVec,0);
            MostTempting_z=max(TemptationMatrix_z,[],1);

            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);

            entireRHS_z=ReturnMatrix_z+TemptationMatrix_z-ones(N_a,1).*MostTempting_z+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);

            z_val=z_gridvals(z_c,:);
            for a_c=1:N_a
                a_val=a_gridvals(a_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec,0);

                TemptationMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(TemptationFn, 0, special_n_a, special_n_z, 0, a_val, z_val, TemptationFnParamsVec,0);
                MostTempting_az=max(TemptationMatrix_az,[],1);

                entireRHS_az=ReturnMatrix_az+TemptationMatrix_az-ones(N_a,1).*MostTempting_az+DiscountFactorParamsVec*EV_z;
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
    TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    if fieldexists_pi_z_J==1
        z_grid=vfoptions.z_grid_J(:,jj);
        pi_z=vfoptions.pi_z_J(:,:,jj);
    elseif fieldexists_ExogShockFn==1
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
            z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        else
            [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
            z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        end
    end
    if vfoptions.lowmemory>0 && (fieldexists_pi_z_J==1 || fieldexists_ExogShockFn==1)
        if all(size(z_grid)==[sum(n_z),1])
            z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
        elseif all(size(z_grid)==[prod(n_z),l_z])
            z_gridvals=z_grid;
        end
    end
    
    VKronNext_j=V(:,:,jj+1);
    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid, ReturnFnParamsVec,0);
        
        TemptationMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(TemptationFn, 0, n_a, n_z, 0, a_grid, z_grid, TemptationFnParamsVec,0);
        MostTempting=max(TemptationMatrix,[],1);

        if vfoptions.paroverz==1
            % (aprime,a,z)
            
            % Use sparse for a few lines until sum over zprime
            EV=VKronNext_j.*shiftdim(pi_z',-1);
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension
            
            entireRHS=ReturnMatrix+TemptationMatrix-ones(N_a,1).*MostTempting+DiscountFactorParamsVec*EV.*ones(1,N_a,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);
            
            V(:,:,jj)=shiftdim(Vtemp,1);
            Policy(:,:,jj)=shiftdim(maxindex,1);
            
        elseif vfoptions.paroverz==0
            
            for z_c=1:N_z
                ReturnMatrix_z=ReturnMatrix(:,:,z_c);
                TemptationMatrix_z=ReturnMatrix(:,:,z_c);
                MostTempting_z=MostTempting(1,:,z_c);
                
                %Calc the condl expectation term (except beta), which depends on z but
                %not on control variables
                EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);
                
                entireRHS_z=ReturnMatrix_z+TemptationMatrix_z-ones(N_a,1).*MostTempting_z+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V(:,z_c,jj)=Vtemp;
                Policy(:,z_c,jj)=maxindex;
            end
        end
        
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, special_n_z, 0, a_grid, z_val, ReturnFnParamsVec,0);
            
            TemptationMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(TemptationFn, 0, n_a, special_n_z, 0, a_grid, z_val, TemptationFnParamsVec,0);
            MostTempting_z=max(TemptationMatrix_z,[],1);

            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            entireRHS_z=ReturnMatrix_z+TemptationMatrix_z-ones(N_a,1).*MostTempting_z+DiscountFactorParamsVec*EV_z*ones(1,N_a,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,jj)=Vtemp;
            Policy(:,z_c,jj)=maxindex;
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
                        
            z_val=z_gridvals(z_c,:);
            for a_c=1:N_a
                a_val=a_gridvals(a_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, special_n_a, special_n_z, 0, a_val, z_val, ReturnFnParamsVec,0);

                TemptationMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(TemptationFn, 0, special_n_a, special_n_z, 0, a_val, z_val, TemptationFnParamsVec,0);
                MostTempting_az=max(TemptationMatrix_az,[],1);
                
                entireRHS_az=ReturnMatrix_az+TemptationMatrix_az-ones(N_a,1).*MostTempting_az+DiscountFactorParamsVec*EV_z;
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_az);
                V(a_c,z_c,jj)=Vtemp;
                Policy(a_c,z_c,jj)=maxindex;
            end
        end
        
    end
end


end