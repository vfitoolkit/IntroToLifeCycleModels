function ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case2_PType(StationaryDist, Policy, FnsToEvaluate, Parameters, n_d, n_a, n_z, N_j, Names_i, d_grid, a_grid, z_grid, simoptions)
% Allows for different permanent (fixed) types of agent. 
% See ValueFnIter_Case1_FHorz_PType for general idea.
%
% simoptions.verbose=1 will give feedback
% simoptions.verboseparams=1 will give further feedback on the param values of each permanent type
%
% jequaloneDist can either be same for all permanent types, or must be passed as a structure.
% AgeWeightParamNames is either same for all permanent types, or must be passed as a structure.
%
%
% How exactly to handle these differences between permanent (fixed) types
% is to some extent left to the user. You can, for example, input
% parameters that differ by permanent type as a vector with different rows f
% for each type, or as a structure with different fields for each type.
%
% Any input that does not depend on the permanent type is just passed in
% exactly the same form as normal.

% Names_i can either be a cell containing the 'names' of the different
% permanent types, or if there are no structures used (just parameters that
% depend on permanent type and inputted as vectors or matrices as appropriate) 
% then Names_i can just be the number of permanent types (but does not have to be, can still be names).
if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i; % It is the number of PTypes (which have not been given names)
    Names_i={'ptype001'};
    for ii=2:N_i
        if ii<10
            Names_i{ii}=['ptype00',num2str(ii)];
        elseif ii<100
            Names_i{ii}=['ptype0',num2str(ii)];
        elseif ii<1000
            Names_i{ii}=['ptype',num2str(ii)];
        end
    end
end

if isstruct(FnsToEvaluate)
    numFnsToEvaluate=length(fieldnames(FnsToEvaluate));
else
    numFnsToEvaluate=length(FnsToEvaluate);
end

% RIGHT NOW THIS ValuesOnGrid ONLY WORKS WHEN ALL AGENTS ARE ON THE SAME GRID
N_a=prod(n_a);
N_z=prod(n_z);
if ~isstruct(FnsToEvaluate)
    if isa(StationaryDist.(Names_i{1}), 'gpuArray')
        ValuesOnDist_Kron=nan(numFnsToEvaluate,N_a,N_z,N_j,'gpuArray');
    else
        ValuesOnDist_Kron=nan(numFnsToEvaluate,N_a,N_z,N_j);
    end
end
ValuesOnGrid=struct();

% Set default of grouping all the PTypes together when reporting statistics
if ~exist('simoptions','var')
    simoptions.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
    simoptions.verbose=0;
    simoptions.verboseparams=0;
else
    if ~isfield(simoptions,'ptypestorecpu')
        simoptions.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
    end
    if ~isfield(simoptions,'verboseparams')
        simoptions.verboseparams=100;
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=100;
    end
end

%%
for ii=1:N_i

    % First set up simoptions
    simoptions_temp=PType_Options(simoptions,Names_i,ii); % Note: already check for existence of simoptions and created it if it was not inputted
    
    if simoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end    

    if simoptions_temp.ptypestorecpu==1 % Things are being stored on cpu but solved on gpu
        PolicyIndexes_temp=gpuArray(Policy.(Names_i{ii}));
        StationaryDist_temp=gpuArray(StationaryDist.(Names_i{ii}));
    else
        PolicyIndexes_temp=Policy.(Names_i{ii});
        StationaryDist_temp=StationaryDist.(Names_i{ii});
    end
    % Parallel is determined by StationaryDist, unless it is specified
    if isa(StationaryDist_temp, 'gpuArray')
        Parallel_temp=2;
    else
        Parallel_temp=1;
    end
    if isfield(simoptions_temp,'parallel')
        Parallel_temp=simoptions.parallel;
        if Parallel_temp~=2
            PolicyIndexes_temp=gather(PolicyIndexes_temp);
            StationaryDist_temp=gather(StationaryDist_temp);
        end
    end
    
    % Go through everything which might be dependent on permanent type (PType)
    % Notice that the way this is coded the grids (etc.) could be either
    % fixed, or a function (that depends on age, and possibly on permanent
    % type), or they could be a structure. Only in the case where they are
    % a structure is there a need to take just a specific part and send
    % only that to the 'non-PType' version of the command.
    if isa(n_d,'struct')
        n_d_temp=n_d.(Names_i{ii});
    else
        n_d_temp=n_d;
    end
    if isa(n_a,'struct')
        n_a_temp=n_a.(Names_i{ii});
    else
        n_a_temp=n_a;
    end
    if isa(n_z,'struct')
        n_z_temp=n_z.(Names_i{ii});
    else
        n_z_temp=n_z;
    end
    if isa(N_j,'struct')
        N_j_temp=N_j.(Names_i{ii});
    else
        N_j_temp=N_j;
    end
    if isa(d_grid,'struct')
        d_grid_temp=d_grid.(Names_i{ii});
    else
        d_grid_temp=d_grid;
    end
    if isa(a_grid,'struct')
        a_grid_temp=a_grid.(Names_i{ii});
    else
        a_grid_temp=a_grid;
    end
    if isa(z_grid,'struct')
        z_grid_temp=z_grid.(Names_i{ii});
    else
        z_grid_temp=z_grid;
    end

    % Parameters are allowed to be given as structure, or as vector/matrix
    % (in terms of their dependence on fixed type). So go through each of
    % these in term.
    Parameters_temp=Parameters;
    FullParamNames=fieldnames(Parameters);
    nFields=length(FullParamNames);
    for kField=1:nFields
        if isa(Parameters.(FullParamNames{kField}), 'struct') % Check for permanent type in structure form
            names=fieldnames(Parameters.(FullParamNames{kField}));
            for jj=1:length(names)
                if strcmp(names{jj},Names_i{ii})
                    Parameters_temp.(FullParamNames{kField})=Parameters.(FullParamNames{kField}).(names{jj});
                end
            end
        elseif any(size(Parameters.(FullParamNames{kField}))==N_i) % Check for permanent type in vector/matrix form.
            temp=Parameters.(FullParamNames{kField});
            [~,ptypedim]=max(size(Parameters.(FullParamNames{kField}))==N_i); % Parameters as vector/matrix can be at most two dimensional, figure out which relates to PType.
            if ptypedim==1
                Parameters_temp.(FullParamNames{kField})=temp(ii,:);
            elseif ptypedim==2
                Parameters_temp.(FullParamNames{kField})=temp(:,ii);
            end
        end
    end
    
    if simoptions_temp.verboseparams==1
        sprintf('Parameter values for the current permanent type')
        Parameters_temp
    end
    

    % Figure out which functions are actually relevant to the present PType. Only the relevant ones need to be evaluated.
    % The dependence of FnsToEvaluate and FnsToEvaluateFnParamNames are necessarily the same.
    % Allows for FnsToEvaluate as structure.
    if n_d_temp(1)==0
        l_d_temp=0;
    else
        l_d_temp=1;
    end
    l_a_temp=length(n_a_temp);
    l_z_temp=length(n_z_temp);  
    [FnsToEvaluate_temp,FnsToEvaluateParamNames_temp, WhichFnsForCurrentPType,~]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0,2);
    
    disp('Here3')
    ValuesOnGrid_ii=EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case2(PolicyIndexes_temp, FnsToEvaluate_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, N_j_temp, d_grid_temp, a_grid_temp, z_grid_temp, Parallel_temp, simoptions_temp);

    if isfield(simoptions_temp,'n_e')
        n_z_temp=[n_z_temp,simoptions_temp.n_e];
    end
    if isstruct(FnsToEvaluate)
        FnNames=fieldnames(FnsToEvaluate);
        for kk=1:numFnsToEvaluate
            jj=WhichFnsForCurrentPType(kk);
            if jj>0
                if simoptions.ptypestorecpu==0
                    ValuesOnGrid.(FnNames{kk}).(Names_i{ii})=reshape(ValuesOnGrid_ii.(FnNames{kk}),[n_a_temp,n_z_temp,N_j_temp]);
                else
                    ValuesOnGrid.(FnNames{kk}).(Names_i{ii})=gather(reshape(ValuesOnGrid_ii.(FnNames{kk}),[n_a_temp,n_z_temp,N_j_temp]));
                end
            end
        end
        
    else % Note: this only works when all agents use same grid
        for kk=1:numFnsToEvaluate
            jj=WhichFnsForCurrentPType(kk);
            if jj>0
                ValuesOnDist_Kron(kk,:,:,:)=ValuesOnGrid_ii(jj,:,:,:);
            end
        end
        if simoptions.ptypestorecpu==0
            ValuesOnGrid.(Names_i{ii})=reshape(ValuesOnDist_Kron,[numFnsToEvaluate,n_a_temp,n_z_temp,N_j_temp]);
        else
            ValuesOnGrid.(Names_i{ii})=gather(reshape(ValuesOnDist_Kron,[numFnsToEvaluate,n_a_temp,n_z_temp,N_j_temp]));
        end
    end
end


end

