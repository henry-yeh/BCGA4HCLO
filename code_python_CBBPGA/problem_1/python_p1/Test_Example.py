# -*- encoding: utf-8 -*-
'''
@File    :   Test_Example.py
@Time    :   2021/12/09 17:17:57
@Author  :   Xianqi
@Contact :   chenxianqi12@qq.com
@Function:   This an example for fitness evaluation.
'''


import numpy as np
from utils import getObjectiveConstraint, plot_layout
from Param import domain, component, heatpipe


def main():

    dim = component.x_opt_dim
    x = np.random.rand(1, dim) * (component.x_opt_max - component.x_opt_min) + component.x_opt_min
    
    objective, constraint = getObjectiveConstraint(x, domain, component, heatpipe)
    obj = objective[0]
    cons1, cons2, cons3, cons4 = constraint
    print()
    print("Objective value: ", obj, "W")
    print("Constraint 1 voilation: ", cons1)
    print("Constraint 2 voilation: ", cons2)
    print("Constraint 3 voilation: ", cons3)
    print("Constraint 4 voilation: ", cons4)

    # draw layout
    plot_layout(x, domain, component, heatpipe, savefig=False, disfig=True, dismass=True, prefix_name='p1')


if __name__ == "__main__":
    main()
