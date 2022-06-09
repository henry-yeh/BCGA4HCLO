# -*- encoding: utf-8 -*-
"""
@File    :   main_test.py
@Time    :   2022/02/21 16:34:45
@Author  :   Xianqi
@Contact :   chenxianqi12@qq.com
"""


import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))


def main():
    problem_type = [4]
    for problem in problem_type:
        if problem == 1:
            problem_path = curPath + "/problem_1/python_p1"
            sys.path.append(problem_path)

            from utils import getObjectiveConstraint
            from Param import domain, component, heatpipe

            sys.path.remove(problem_path)
        elif problem == 2:
            problem_path = curPath + "/problem_2/python_p2"
            sys.path.append(problem_path)

            from utils import getObjectiveConstraint, prepare_data

            json_file = curPath + "/problem_2/python_p2/Problem2.json"
            domain, component, heatpipe = prepare_data(json_file)

            sys.path.remove(problem_path)
        elif problem == 3:
            problem_path = curPath + "/problem_3/python_p3"
            sys.path.append(problem_path)
            from utils import getObjectiveConstraint, prepare_data

            json_file = curPath + "/problem_3/python_p3/Problem3.json"
            domain, component, heatpipe = prepare_data(json_file)

            sys.path.remove(problem_path)
        elif problem == 4:
            problem_path = curPath + "/problem_4/python_p4"
            sys.path.append(problem_path)

            from utils import getObjectiveConstraint, prepare_data

            json_file = curPath + "/problem_4/python_p4/Problem4.json"
            domain, component, heatpipe = prepare_data(json_file)

            sys.path.remove(problem_path)
        else:
            pass

        D = component.x_opt_dim  # obtain the number of design variables
        runtimes = 30
        totalFes = 500 * D  # set the maximum number of function evaluations

        # import your test algorithm coded in file: "example.py"
        # 'main_algorithm': the name of the main function
        from BCGA import main_algorithm

        for _ in range(runtimes):
            # The output should record:
            # 1. the best solution
            # 2. the best solution in each iteration
            # 3. the best objective in each iteration
            bestSolution, iter_best_sol, iter_best_val = main_algorithm(
                totalFes, domain, component, heatpipe, getObjectiveConstraint,
            )


if __name__ == "__main__":
    main()
