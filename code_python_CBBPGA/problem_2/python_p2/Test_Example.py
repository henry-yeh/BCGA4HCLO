# -*- encoding: utf-8 -*-
'''
@File    :   Test_Example.py
@Time    :   2021/12/16 21:38:20
@Author  :   Xianqi
@Contact :   chenxianqi12@qq.com
@Function:   This is an example for fitness evaluation.
'''


import numpy as np
from utils import getObjectiveConstraint, plot_layout, prepare_data


def main(flag="randomly"):
    json_file = "Problem2.json"
    domain, component, heatpipe = prepare_data(json_file)

    if flag == "randomly":
        x_dim = component.x_opt_dim  # x_dim: dimension of design variables
        x_max, x_min = component.x_opt_max, component.x_opt_min  # ranges of design variables [size: 1*x_dim]
        x = x_min + np.random.rand(x_min.shape[0], x_min.shape[1]) * (x_max - x_min)  # randomly generate one layout

        objective, constraint = getObjectiveConstraint(x, domain, component, heatpipe)
    else:  # display the example layout
        x = component.location.reshape(1, -1)
        objective, constraint = getObjectiveConstraint(x, domain, component, heatpipe)

    obj = objective[0]
    cons1, cons2, cons3, cons4 = constraint
    print()
    print("Objective value: ", obj, "W")  # the objective should be minimized.
    print("Constraint 1 voilation (non-overlapping)    : ", cons1)  # all constraint voilations should be zero.
    print("Constraint 2 voilation (centroid range)     : ", cons2)
    print("Constraint 3 voilation (maximum load)       : ", cons3)
    print("Constraint 4 voilation (comp-hp overlapping): ", cons4)

    # draw layout
    plot_layout(x, domain, component, heatpipe, savefig=True, disfig=False, dismass=True, prefix_name="p2.1")


if __name__ == "__main__":
    # randomly generate one layout scheme
    # main()

    # display the example layout (feasible)
    main(flag=1)
