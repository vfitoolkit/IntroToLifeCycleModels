function ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_Case2(PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel, simoptions, EntryExitParamNames, StationaryDist)
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate
%
% Parallel, simoptions and EntryExitParamNames are optional inputs, only needed when using endogenous entry

if exist('Parallel','var')==0
    Parallel=1+(gpuDeviceCount>0);
elseif isempty(Parallel)
    Parallel=1+(gpuDeviceCount>0);
end

if isstruct(StationaryDist)
    % Even though Mass is unimportant, still need to deal with 'exit' in PolicyIndexes.
    % NOTE: THIS DOES NOT EXIST SO WILL THROW AN ERROR
    ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_Case2_Mass(StationaryDist.mass, PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames,EntryExitParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel,simoptions);
    return
end

if Parallel==2 || Parallel==4
    Parallel=2;
    PolicyIndexes=gpuArray(PolicyIndexes);
    n_d=gpuArray(n_d);
    n_a=gpuArray(n_a);
    n_z=gpuArray(n_z);
    d_grid=gpuArray(d_grid);
    a_grid=gpuArray(a_grid);
    z_grid=gpuArray(z_grid);
    
    % l_d not needed with Parallel=2 implementation
    l_a=length(n_a);
    l_z=length(n_z);
    
    N_a=prod(n_a);
    N_z=prod(n_z);
    
    ValuesOnGrid=zeros(N_a*N_z,length(FnsToEvaluate),'gpuArray');
    
    PolicyValues=PolicyInd2Val_Case2(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid, Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];    
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,l_d]
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names)  % check for 'SSvalueParamNames={}'
            FnToEvaluateParamsCell=[];
        else
            FnToEvaluateParamsCell=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
        end
        Values=EvalFnOnAgentDist_Grid_Case2(FnsToEvaluate{i}, FnToEvaluateParamsCell,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values=reshape(Values,[N_a*N_z,1]);
%         ProbDensityFns(:,i)=Values.*StationaryDistVec;
        ValuesOnGrid(:,i)=Values;
    end
    
else
    l_d=length(n_d);
    
    N_a=prod(n_a);
    N_z=prod(n_z);
    
    [d_gridvals, ~]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,2, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);    
    
    ValuesOnGrid=zeros(N_a*N_z,length(FnsToEvaluate));
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names) % check for 'SSvalueParamNames={}'
            Values=zeros(N_a*N_z,1);
            for ii=1:N_a*N_z
                %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
            end
            ValuesOnGrid(:,i)=Values;
        else
            FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
            Values=zeros(N_a*N_z,1);
            for ii=1:N_a*N_z
                %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
            end
            ValuesOnGrid(:,i)=Values;
        end
    end
end

% Change the ordering and size so that ProbDensityFns has same kind of
% shape as StationaryDist, except first dimension indexes the
% 'FnsToEvaluate'.
ValuesOnGrid=ValuesOnGrid';
ValuesOnGrid=reshape(ValuesOnGrid,[length(FnsToEvaluate),n_a,n_z]);

end
