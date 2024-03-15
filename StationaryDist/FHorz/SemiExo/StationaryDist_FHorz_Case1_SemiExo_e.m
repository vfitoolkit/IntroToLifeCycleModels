function StationaryDist=StationaryDist_FHorz_Case1_SemiExo_e(jequaloneDistKron,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,pi_e_J,Parameters,simoptions)

if n_z(1)==0
    error('Not yet implemented n_z=0 with SemiExo, email me and I will do it (or you can just pretend by using n_z=1 and pi_z=1, not using the value of z anywhere)')
end

%% Setup related to semi-exogenous state (an exogenous state whose transition probabilities depend on a decision variable)
if ~isfield(simoptions,'n_semiz')
    error('When using simoptions.SemiExoShockFn you must declare simoptions.n_semiz')
end
if ~isfield(simoptions,'semiz_grid')
    error('When using simoptions.SemiExoShockFn you must declare simoptions.semiz_grid')
end
if ~isfield(simoptions,'d_grid')
    error('When using simoptions.SemiExoShockFn you must declare simoptions.d_grid')
else
    simoptions.d_grid=gpuArray(simoptions.d_grid);
end
if ~isfield(simoptions,'numd_semiz')
    simoptions.numd_semiz=1; % by default, only one decision variable influences the semi-exogenous state
end
if length(n_d)>simoptions.numd_semiz
    n_d1=n_d(1:end-simoptions.numd_semiz);
    d1_grid=simoptions.d_grid(1:sum(n_d1));
else
    n_d1=0; d1_grid=[];
end
n_d2=n_d(end-simoptions.numd_semiz+1:end); % n_d2 is the decision variable that influences the transition probabilities of the semi-exogenous state
l_d2=length(n_d2);
d2_grid=simoptions.d_grid(sum(n_d1)+1:end);
% Create the transition matrix in terms of (d,zprime,z) for the semi-exogenous states for each age
N_semiz=prod(simoptions.n_semiz);
l_semiz=length(simoptions.n_semiz);
temp=getAnonymousFnInputNames(simoptions.SemiExoStateFn);
if length(temp)>(l_semiz+l_semiz+l_d2) % This is largely pointless, the SemiExoShockFn is always going to have some parameters
    SemiExoStateFnParamNames={temp{l_semiz+l_semiz+l_d2+1:end}}; % the first inputs will always be (semiz,semizprime,d)
else
    SemiExoStateFnParamNames={};
end
pi_semiz_J=zeros(N_semiz,N_semiz,prod(n_d2),N_j);
for jj=1:N_j
    SemiExoStateFnParamValues=CreateVectorFromParams(Parameters,SemiExoStateFnParamNames,jj);
    pi_semiz_J(:,:,:,jj)=CreatePiSemiZ(n_d2,simoptions.n_semiz,d2_grid,simoptions.semiz_grid,simoptions.SemiExoStateFn,SemiExoStateFnParamValues);
end


%%
n_e=simoptions.n_e;

N_d1=prod(n_d1);
% N_d2=prod(n_d2);
N_a=prod(n_a);
N_z=prod(n_z);
% N_semiz=prod(n_semiz);
N_e=prod(n_e);

if ~isfield(simoptions,'loopovere')
    simoptions.loopovere=0; % default is parallel over e, 1 will loop over e, 2 will parfor loop over e
end

%%
jequaloneDistKron=reshape(jequaloneDistKron,[N_a*N_semiz*N_z*N_e,1]);
Policy=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, [simoptions.n_semiz,n_z],N_j,n_e);
if simoptions.iterate==0
    Policy=gather(Policy);
    jequaloneDistKron=gather(jequaloneDistKron);    
end
pi_z_J=gather(pi_z_J);
pi_e_J=gather(pi_e_J);

if simoptions.iterate==0
    StationaryDist=StationaryDist_FHorz_Case1_SemiExo_Simulation_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy,n_d1,n_d2,N_a,N_z,N_semiz,N_e,N_j,pi_z_J,pi_semiz_J,pi_e_J,Parameters,simoptions);
elseif simoptions.iterate==1
    StationaryDist=StationaryDist_FHorz_Case1_SemiExo_Iteration_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy,N_d1,N_a,N_z,N_semiz,N_e,N_j,pi_z_J,pi_semiz_J,pi_e_J,Parameters,simoptions);
end

if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist); % move output to gpu
end
if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,simoptions.n_semiz,n_z,n_e,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_semiz,N_z,N_e,N_j]);
end

end
