function prob=LifeCycleModelA9_SemiExoStateFn(n,work,nprime,workprime,f,search,probofbirth,probofadult,probjobseperation)

probn=-1;  % Just a placeholder (one that will cause errors if not overwritten)
probwork=-1;  % Just a placeholder (one that will cause errors if not overwritten)

% First, probabilites for workprime, which depend on work and search
if work==1 
    if workprime==0
        probwork=probjobseperation; % exogenous probability of losing job
    elseif workprime==1
        probwork=1-probjobseperation;
    end
elseif work==0
    if workprime==0
        probwork=1-search;
    elseif workprime==1
        probwork=search; % search is 0 to 1, and is probability you find job
    end
end

% Second, the probabilities for nprime, which depend on n and f
% Call these probn
if n==0
    if f==0
        if nprime==0
            probn=1; % If dont have infant and f=0 then definitely no infant;
        elseif nprime==1
            probn=0;
        elseif nprime==2
            probn=0;
        end
    elseif f==1
        if nprime==0
            probn=1-probofbirth;
        elseif nprime==1
            probn=probofbirth;
        elseif nprime==2
            probn=0;
        end
    end
elseif n==1 % Note, this is effectively independent of f [Note that if model is annual, then it is not possible to have infants in consecutive years]
    if f==0
        if nprime==0
            probn=probofadult; % Child becomes and adult (so -1 child)
        elseif nprime==1
            probn=1-probofadult;
        elseif nprime==2
            probn=0;
        end
    elseif f==1
        if nprime==0
            probn=probofadult*(1-probofbirth); % Child becomes and adult (so -1 child), and no newborn
        elseif nprime==1
            probn=(1-probofadult)*(1-probofbirth)+probofadult*probofbirth; % either child stays and no newborn, or child becomes adult and a newborn
        elseif nprime==2
            probn=(1-probofadult)*probofbirth; % child says and a newborn
        end
    end
elseif n==2 % 2 is max number of children (because of our grid), so cannot try having child if you already have 2
    if nprime==0
        probn=0;
    elseif nprime==1
        probn=probofadult; % Child becomes and adult (so -1 child)
    elseif nprime==2
        probn=1-probofadult;
    end
end

prob=probn*probwork;



end
