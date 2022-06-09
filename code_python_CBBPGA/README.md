We propose a bi-population clan-based genetic algorithm (BCGA). 

A clan-based framework featuring no extra computational cost or sensitivity to pre-defined parameters, is designed to increase population diversity and locate different optimal regions, where a clan is defined to be a production and competition unit with a dynamically increasing clan size. The framework introduces special rules to adjust both the selection and crossover operators to preserve population diversity and boost the exploration ability of the algorithm.

A bi-population strategy is proposed to decompose the problem and unleash the evolution potential of the GA population. The problem is decomposed into two sub-problems: finding a satisfying component distribution along the x-axis and finding a feasible component distribution along the y-axis that tailored for the best distribution along the x-axis. Hence, We divide the whole population into two subpopulations. Subpopulation1, serves to solve sub-problem1 under the clan-based framework, while subpopulation2 serves to solve sub-problem2.

SBX is adopted as the crossover operator. The mutation operator combines PLM and a basic swap mutation. The latter randomly switches the values of an individual's two design variables one or more times. The binary tournament selection without replacement is utilized based on two specially designed cost functions for two subpopulations, respectively.

Run main_test.py to test our algorithm.


 
