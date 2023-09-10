function F=LifeCycleModel38_ReturnFn(aprime,a,clag,z,w,sigma,agej,Jr,pension,r,kappa_j,lambda,mu,upsilon,theta)

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*z+(1+r)*a-aprime; % Add z here
else % Retirement
    c=pension+(1+r)*a-aprime;
end

if c>0
    u_c=(c^(1-sigma))/(1-sigma); % The utility function
    u_clag=(clag^(1-sigma))/(1-sigma); % The utility function
    Delta=u_c-u_clag;
    if c>=clag % so u_c>=u_clag, Delta>=0
        v=(1-exp(-mu*Delta))/mu;
    else % c<clag, so u_c<u_clag, Delta<0
        v=-lambda*(1-exp((upsilon/lambda)*Delta))/upsilon;
    end
    F=theta*u_c+(1-theta)*v;
end

% % To test, I make sure clag does nothing if using a basic setup
% if c>0
%     F=(c^(1-sigma))/(1-sigma); % The utility function
% end


end
