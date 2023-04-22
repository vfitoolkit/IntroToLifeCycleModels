function prob=SemiExoStateFn(n1,n2,n1prime,n2prime,f,probofbirth,probofchild,probofadult)

prob=-1; % Just a placeholder (one that will cause errors if not overwritten)
probn1=-1;
probn2=Inf;

% First, the probabilities for n2prime, which depend on n1 and n2 but not on f
% Call these probn2
if n1==0
    if n2==0
        if n2prime==0
            probn2=1;
        elseif n2prime==1
            probn2=0;
        elseif n2prime==2
            probn2=0;
        elseif n2prime==3
            probn2=0;
        end
    elseif n2==1
        if n2prime==0
            probn2=probofadult;
        elseif n2prime==1
            probn2=1-probofadult;
        elseif n2prime==2
            probn2=0;
        elseif n2prime==3
            probn2=0;
        end
    elseif n2==2
        if n2prime==0
            probn2=0;
        elseif n2prime==1
            probn2=probofadult;
        elseif n2prime==2
            probn2=1-probofadult;
        elseif n2prime==3
            probn2=0;
        end
    elseif n2==3
        if n2prime==0
            probn2=0;
        elseif n2prime==1
            probn2=0;
        elseif n2prime==2
            probn2=probofadult;
        elseif n2prime==3
            probn2=1-probofadult;
        end
    end
elseif n1==1
    if n2==0
        if n2prime==0
            probn2=1-probofchild;
        elseif n2prime==1
            probn2=probofchild;
        elseif n2prime==2
            probn2=0;
        elseif n2prime==3
            probn2=0;
        end
    elseif n2==1
        if n2prime==0
            probn2=probofadult*(1-probofchild);
        elseif n2prime==1
            probn2=(1-probofadult)*(1-probofchild)+probofchild*probofadult;
        elseif n2prime==2
            probn2=(1-probofadult)*probofchild;
        elseif n2prime==3
            probn2=0;
        end
    elseif n2==2
        if n2prime==0
            probn2=0;
        elseif n2prime==1
            probn2=probofadult*(1-probofchild);
        elseif n2prime==2
            probn2=(1-probofadult)*(1-probofchild)+probofchild*probofadult;
        elseif n2prime==3
            probn2=(1-probofadult)*probofchild;
        end
    elseif n2==3
        if n2prime==0
            probn2=0;
        elseif n2prime==1
            probn2=0;
        elseif n2prime==2
            probn2=probofadult*(1-probofchild);
        elseif n2prime==3
            probn2=(1-probofadult)*(1-probofchild)+probofchild; % Note: It is just assumed that if n2=3 and n1=1, and the infant becomes a child, then one of the children is forced to become an adult
        end
    end
end

% Second, the probabilities for n1prime, which depend on n1 and f
% Call these probn1
if n1==0
    if f==0
        if n1prime==0
            probn1=1; % If dont have infant and f=0 then definitely no infant;
        elseif n1prime==1
            probn1=0;
        end
    elseif f==1
        if n1==0
            if n1prime==0
                probn1=1-probofbirth;
            elseif n1prime==1
                probn1=probofbirth;
            end
        end
    end
end
if n1==1 % Note, this is effectively independent of f [Note that if model is annual, then it is not possible to have infants in consecutive years]
    if n1prime==0
        probn1=probofchild; % If dont have infant and f=0 then definitely no infant;
    elseif n1prime==1
        probn1=1-probofchild;
    end
end

prob=probn1*probn2;



end