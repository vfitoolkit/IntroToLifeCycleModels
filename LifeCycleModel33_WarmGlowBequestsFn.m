function W=LifeCycleModel33_WarmGlowBequestsFn(aprime,wg,sigma,agej,J) 

W=0;

if agej==J
    W=-Inf;
    if aprime>0
        W=wg*(aprime^(1-sigma))/(1-sigma);
    end
end

end