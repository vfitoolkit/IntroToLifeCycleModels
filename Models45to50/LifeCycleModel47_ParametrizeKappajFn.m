function Params=LifeCycleModel47_ParametrizeKappajFn(Params)

agevec=(Params.agejshifter+Params.agej)/100; % real world ages (this just seems a nicer thing to put the polynomial on, rather than model period; easier to relate to data work)

% log(kappa_j) is a fifth-order polynomial in age (actually, age divided by 100, as otherwise the polynomial just gives 
% numbers so large that when we take the exponential they blow up)
Params.kappa_j=exp(Params.kappa_j_c0+Params.kappa_j_c1*agevec+Params.kappa_j_c2*agevec.^2+Params.kappa_j_c3*agevec.^3+Params.kappa_j_c4*agevec.^4+Params.kappa_j_c5*agevec.^5);
% Note: in the estimation we require that kappa_j is positive, and by
% making log(kappa_j) a polynomial, rather than making kappa_j a
% polynomial, we ensure that kappa_j is alway positive.

end