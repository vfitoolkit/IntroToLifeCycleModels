function [V,Policy2]=ValueFnIter_Case1_FHorz_GulPesendorfer_e_raw(n_d,n_a,n_z,n_e,N_j, d_grid, a_grid, z_grid, e_grid,pi_z, pi_e, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

if n_z(1)==0
    l_z=0;
else
    l_z=length(n_z);
end

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
z_grid=gpuArray(z_grid);
e_grid=gpuArray(e_grid);

eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')

eval('fieldexists_EiidShockFn=1;vfoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
eval('fieldexists_EiidShockFnParamNames=1;vfoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
eval('fieldexists_pi_e_J=1;vfoptions.pi_e_J;','fieldexists_pi_e_J=0;')


if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
    % e_gridvals is created below
end
if vfoptions.lowmemory>1
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
    % z_gridvals is created below
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
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
if fieldexists_pi_e_J==1
    e_grid=vfoptions.e_grid_J(:,N_j);
    pi_e=vfoptions.pi_e_J(:,N_j);
elseif fieldexists_EiidShockFn==1
    if fieldexists_EiidShockFnParamNames==1
        EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,N_j);
        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
        for ii=1:length(EiidShockFnParamsVec)
            EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
        end
        [e_grid,pi_e]=vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
        e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
    else
        [e_grid,pi_e]=vfoptions.ExogShockFn(N_j);
        e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
    end
end

pi_e=shiftdim(pi_e,-2); % Move to third dimension

if vfoptions.lowmemory>0
    if all(size(z_grid)==[sum(n_z),1])
        z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(z_grid)==[prod(n_z),l_z])
        z_gridvals=z_grid;
    end
    if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
        e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
        e_gridvals=e_grid;
    end
end

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_grid, e_grid, ReturnFnParamsVec);

        TemptationMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_grid, e_grid, TemptationFnParamsVec);
        MostTempting=max(TemptationMatrix,[],1);
        entireRHS=ReturnMatrix+TemptationMatrix-ones(N_d*N_a,1).*MostTempting;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,:,N_j)=Vtemp;
        Policy(:,:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_grid, e_val, ReturnFnParamsVec);

            TemptationMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_grid, e_val, TemptationFnParamsVec);
            MostTempting_e=max(TemptationMatrix_e,[],1);
            entireRHS_e=ReturnMatrix_e+TemptationMatrix_e-ones(N_d*N_a,1).*MostTempting_e;

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            Policy(:,:,e_c,N_j)=maxindex;
        end

    elseif vfoptions.lowmemory==2

        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            for z_c=1:N_z
                z_val=z_gridvals(z_c,:);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);

                TemptationMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, TemptationFnParamsVec);
                MostTempting_ze=max(TemptationMatrix_ze,[],1);
                entireRHS_ze=ReturnMatrix_ze+TemptationMatrix_ze-ones(N_d*N_a,1).*MostTempting_ze;

                % Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                Policy(:,z_c,e_c,N_j)=maxindex;
            end
        end

    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    V_Jplus1=sum(V_Jplus1.*pi_e,3);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_grid, e_grid, ReturnFnParamsVec);
        % (d,aprime,a,z,e)
        
        EV=V_Jplus1.*shiftdim(pi_z',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        entireEV=repelem(EV,N_d,1,1);

        TemptationMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_grid, e_grid, TemptationFnParamsVec);
        MostTempting=max(TemptationMatrix,[],1);
        entireRHS=ReturnMatrix+TemptationMatrix-ones(N_d*N_a,1).*MostTempting+DiscountFactorParamsVec*repmat(entireEV,1,N_a,1,N_e);
        
        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        
        V(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,:,N_j)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        EV=V_Jplus1.*shiftdim(pi_z',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        entireEV=repelem(EV,N_d,1,1);
        
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_grid, e_val, ReturnFnParamsVec);
            % (d,aprime,a,z)

            TemptationMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_grid, e_val, TemptationFnParamsVec);
            MostTempting_e=max(TemptationMatrix_e,[],1);
            entireRHS_e=ReturnMatrix_e+TemptationMatrix_e-ones(N_d*N_a,1).*MostTempting_e+DiscountFactorParamsVec*entireEV.*ones(1,N_a,1);
                        
            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            
            V(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,N_j)=shiftdim(maxindex,1);
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            
            %Calc the condl expectation term (except beta) which depends on z but not control variables
            EV_z=V_Jplus1.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            entireEV_z=kron(EV_z,ones(N_d,1));

            for e_c=1:N_e
                e_val=e_gridvals(e_c,:);
                
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);
                
                TemptationMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, TemptationFnParamsVec);
                MostTempting_ze=max(TemptationMatrix_ze,[],1);
                entireRHS_ze=ReturnMatrix_ze+TemptationMatrix_ze-ones(N_d*N_a,1).*MostTempting_ze+DiscountFactorParamsVec*entireEV_z*ones(1,N_a,1);
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                Policy(:,z_c,e_c,N_j)=maxindex;
            end
        end
    end
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames, jj);
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
    if fieldexists_pi_e_J==1
        e_grid=vfoptions.e_grid_J(:,jj);
        pi_e=vfoptions.pi_e_J(:,jj);
        pi_e=shiftdim(pi_e,-2); % Move to thrid dimension
    elseif fieldexists_EiidShockFn==1
        if fieldexists_EiidShockFnParamNames==1
            EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,jj);
            EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
            for ii=1:length(EiidShockFnParamsVec)
                EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
            end
            [e_grid,pi_e]=vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
            e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
        else
            [e_grid,pi_e]=vfoptions.EiidShockFn(jj);
            e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
        end
        pi_e=shiftdim(pi_e,-2); % Move to third dimension
    end
    
    if vfoptions.lowmemory>0
        if (vfoptions.paroverz==1 || vfoptions.lowmemory==2) && (fieldexists_pi_z_J==1 || fieldexists_ExogShockFn==1)
            if all(size(z_grid)==[sum(n_z),1])
                z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
            elseif all(size(z_grid)==[prod(n_z),l_z])
                z_gridvals=z_grid;
            end
        end
        if (fieldexists_pi_e_J==1 || fieldexists_EiidShockFn==1)
            if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
                e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
            elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
                e_gridvals=e_grid;
            end
        end
    end
    
    VKronNext_j=V(:,:,:,jj+1);
        
    VKronNext_j=sum(VKronNext_j.*pi_e,3);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_grid, e_grid, ReturnFnParamsVec);
        % (d,aprime,a,z,e)
        
        EV=VKronNext_j.*shiftdim(pi_z',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        entireEV=repelem(EV,N_d,1,1);

        TemptationMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_grid, e_grid, TemptationFnParamsVec);
        MostTempting=max(TemptationMatrix,[],1);
        entireRHS=ReturnMatrix+TemptationMatrix-ones(N_d*N_a,1).*MostTempting+DiscountFactorParamsVec*repmat(entireEV,1,N_a,1,N_e);
        
        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        
        V(:,:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,:,jj)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        EV=VKronNext_j.*shiftdim(pi_z',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        entireEV=repelem(EV,N_d,1,1);
        
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_grid, e_val, ReturnFnParamsVec);
            % (d,aprime,a,z)
            
            TemptationMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_grid, e_val, TemptationFnParamsVec);
            MostTempting_e=max(TemptationMatrix_e,[],1);
            entireRHS_e=ReturnMatrix_e+TemptationMatrix_e-ones(N_d*N_a,1).*MostTempting_e+DiscountFactorParamsVec*entireEV.*ones(1,N_a,1);
            
            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            
            V(:,:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,jj)=shiftdim(maxindex,1);
        end
        
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            
            %Calc the condl expectation term (except beta) which depends on z but not control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            entireEV_z=kron(EV_z,ones(N_d,1));

            for e_c=1:N_e
                e_val=e_gridvals(e_c,:);
                
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);
                
                TemptationMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(TemptationFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, TemptationFnParamsVec);
                MostTempting_ze=max(TemptationMatrix_ze,[],1);
                entireRHS_ze=ReturnMatrix_ze+TemptationMatrix_ze-ones(N_d*N_a,1).*MostTempting_ze+DiscountFactorParamsVec*entireEV_z*ones(1,N_a,1);
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                V(:,z_c,e_c,jj)=Vtemp;
                Policy(:,z_c,e_c,jj)=maxindex;
            end
        end
    end

end

%%
Policy2=zeros(2,N_a,N_z,N_e,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end