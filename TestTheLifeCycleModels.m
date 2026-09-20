% Run to test all the Life-Cycle Models are working without error.

%% Which parts to run
% 1: Models 1-10 - basic
% 2: Models 11-18 - more basics     
% 3: Models 20-30 - shocks and permanent types
% 4: Models 31-35, 35Balt - portfolio choice
% 5: Models 36-39, 36B, 37B, 39B - alternative preferences
% 6: Models 40-44 - two states
% 7: Models 45-50 - GMM estimation
% 8: Models A1-A12
% 9: Assignments
doPart=[1,1,1,1,1,1,1,1,1];

%% Diary of the command window output
if exist('./TestLifeCycleDiary.txt','file')
    delete('./TestLifeCycleDiary.txt') % otherwise diary just appends to the previous run
end
diary ./TestLifeCycleDiary.txt

addpath('./Models20to30/')
addpath('./Models31to35/')
addpath('./Models36to39/')
addpath('./Models40to44/')
addpath('./Models45to50/')
addpath('./ModelsAppendixA/')
addpath('./Assignments/')

%% Part 1: Models 1-10
if doPart(1)==1

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 1: Consumption-Leisure \n')
    LifeCycleModel1

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 2: Retirement \n')
    LifeCycleModel2

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 3: Assets \n')
    LifeCycleModel3

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 4: Life-Cycle Profiles \n')
    LifeCycleModel4

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 5: Earnings are hump-shaped (and explain agent distribution) \n')
    LifeCycleModel5

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 6: Chance of dying \n')
    LifeCycleModel6

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 7: Warm-glow of bequests \n')
    LifeCycleModel7

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 8: Idiosyncratic shocks and heterogeneity (and explain agent distribution) \n')
    LifeCycleModel8

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 9: Idiosyncratic shocks again, AR(1) \n')
    LifeCycleModel9

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 10: Exogenous labor supply \n')
    LifeCycleModel10

end

%% Part 2: Models 11-18
if doPart(2)==1

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 11: Idiosyncratic shocks again, persistent and transitory \n')
    LifeCycleModel11

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 12: Epstein-Zin preferences \n')
    LifeCycleModel12

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 13: Simulating Panel Data \n')
    LifeCycleModel13

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 14: More Life-Cycle Profiles \n')
    LifeCycleModel14

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 15: Consumption and Borrowing Constraints 1 \n')
    LifeCycleModel15

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 16: Consumption and Borrowing Constraints 2 \n')
    LifeCycleModel16

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 17: Precautionary Savings (with exogenous earnings) \n')
    LifeCycleModel17

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 18: Precautionary Savings with Endogenous labor \n')
    LifeCycleModel18

    % LifeCycleModel19 is only equations/explanation, no code

end

%% Part 3: Models 20-30
if doPart(3)==1

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 20: Idiosyncratic shocks that depend on age \n')
    LifeCycleModel20

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 21: Idiosyncratic medical shocks in retirement \n')
    LifeCycleModel21

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 22: Deterministic Economic/productivity growth \n')
    LifeCycleModel22

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 23: Permanent Types: Solving fixed-types \n')
    LifeCycleModel23

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 24: Using Names for Permanent Types: patient and impatient \n')
    LifeCycleModel24

    % There is no model 25 (yet)
    % clearvars -except doPart
    % fprintf('Now solving Life-Cycle Model 25: More Permanent Types \n')
    % LifeCycleModel25

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 26: Earnings Dynamics (Gaussian-Mixtures) \n')
    LifeCycleModel26

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 27: Two decision variables (dual-earner household) \n')
    LifeCycleModel27

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 28: Semi-exogenous state (fertility and children) \n')
    LifeCycleModel28

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 29: Exploiting Monotonicity for Faster Solutions (Divide-and-Conquer) \n')
    LifeCycleModel29

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 30: Linear Interpolation of aprime (Grid Interpolation Layer) \n')
    LifeCycleModel30

end

%% Part 4: Models 31-35, 35Balt
if doPart(4)==1

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 31: Portfolio-Choice \n')
    LifeCycleModel31

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 32: Portfolio-Choice with Epstein-Zin preferences \n')
    LifeCycleModel32

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 33: Portfolio-Choice with Warm-Glow of Bequests \n')
    LifeCycleModel33

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 34: Portfolio-Choice with Endogenous Labor Supply \n')
    LifeCycleModel34

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 35: Portfolio-Choice with Housing \n')
    LifeCycleModel35

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 35Balt: Portfolio-Choice with Housing, without Epstein-Zin preferences \n')
    LifeCycleModel35Balt

end

%% Part 5: Models 36-39, 36B, 37B, 39B
if doPart(5)==1

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 36: Impatience (Quasi-Hyperbolic Discounting) \n')
    LifeCycleModel36

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 36B: Quasi-Hyperbolic Discounting with Endogenous Labor Supply \n')
    LifeCycleModel36B

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 37: Temptation and Self-Control (Gul-Pesendorfer Preferences) \n')
    LifeCycleModel37

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 37B: Gul-Pesendorfer Preferences with Endogenous Labor \n')
    LifeCycleModel37B

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 38: Loss Aversion (Prospect Theory) \n')
    LifeCycleModel38

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 39: Ambiguity Aversion \n')
    LifeCycleModel39

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 39B: Ambiguity Aversion with Endogenous Labor \n')
    LifeCycleModel39B

end

%% Part 6: Models 40-44
if doPart(6)==1

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 40: Two Endogenous States (Housing) \n')
    LifeCycleModel40

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 41: Female Labor Force Participation History (experienceasset) \n')
    LifeCycleModel41

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 42: Uncertain Human Capital (experienceassetu) \n')
    LifeCycleModel42

    % Models 43 and 44 are not finished yet (their subsections in the pdf are
    % still commented out), so leave them out of the test for now
    % clearvars -except doPart
    % fprintf('Now solving Life-Cycle Model 43: Pensions based on Lifetime Earnings (experienceassetz) \n')
    % LifeCycleModel43

    % clearvars -except doPart
    % fprintf('Now solving Life-Cycle Model 44: Liquid and Illiquid Assets \n')
    % LifeCycleModel44

end

%% Part 7: Models 45-50
% Temporarily running only Model 46, while debugging it; the other five are
% commented out below and doPart is set to run only this part
if doPart(7)==1

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 45: GMM Estimation, the basics \n')
    LifeCycleModel45

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 46: GMM Estimation, parameter restrictions, estimating shocks and initial dist \n')
    LifeCycleModel46

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 47: GMM Estimation, using data and how to choose Weighting Matrix \n')
    LifeCycleModel47

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 48: GMM Estimation, various extras \n')
    LifeCycleModel48

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 49: GMM Estimation, permanent types as unobserved heterogeneity \n')
    LifeCycleModel49

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model 50: GMM Estimation, permanent types 2 \n')
    LifeCycleModel50

end

%% Part 8: Models A1-A12
% Now the appendix models
if doPart(8)==1

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model A1: AR(1) process, alternative quadrature methods \n')
    LifeCycleModelA1

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model A2: AR(1) with gaussian-mixture innovations \n')
    LifeCycleModelA2

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model A3: Permanent (unit-root/random-walk) shocks \n')
    LifeCycleModelA3

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model A4: Age-dependent shocks \n')
    LifeCycleModelA4

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model A5i: Two markov exogenous states, z \n')
    LifeCycleModelA5i

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model A5ii: Two markov (z) shocks, joint-grids \n')
    LifeCycleModelA5ii

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model A5iii: Joint-grids for Correlated Shocks, two markov z \n')
    LifeCycleModelA5iii

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model A5iv: Two markov (z) shocks, probabilities that depend on each other \n')
    LifeCycleModelA5iv

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model A6: Two i.i.d. (e) shocks \n')
    LifeCycleModelA6

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model A7: Three markov (z) and three iid (e) shocks \n')
    LifeCycleModelA7

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model A8: VAR(1) persistent shocks \n')
    LifeCycleModelA8

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model A9: Two age-dependent semi-exogenous (semiz) shocks \n')
    LifeCycleModelA9

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model A10: Second-Order Markov Processes (implementing an AR(2) persistent shock) \n')
    LifeCycleModelA10

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model A11: Model an i.i.d. as an markov exogenous state \n')
    LifeCycleModelA11

    clearvars -except doPart
    fprintf('Now solving Life-Cycle Model A12: Parametrize exogenous shocks - ExogShockFn and EiidShockFn \n')
    LifeCycleModelA12

end

%% Part 9: Assignments
% Now the assignment models
if doPart(9)==1

    clearvars -except doPart
    fprintf('Now solving Assignment 1 Life-Cycle Model: Add a tax on labor income \n')
    Assignment1_LifeCycleModel

    clearvars -except doPart
    fprintf('Now solving Assignment 2 Life-Cycle Model: Alternative Utility Functions \n')
    Assignment2_LifeCycleModel

    % Assignment3_LifeCycleModel.m does not exist yet, so this stays commented out
    % clearvars -except doPart
    % fprintf('Now solving Assignment 3 Life-Cycle Model \n')
    % Assignment3_LifeCycleModel

    clearvars -except doPart
    fprintf('Now solving Assignment 4 Life-Cycle Model: Deterministic income growth with exogenous labor supply and exogenous shocks \n')
    Assignment4_LifeCycleModel

end

fprintf('Finished Life-Cycle Models \n')

diary off
