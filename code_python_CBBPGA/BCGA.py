import copy
import math
import numpy as np
import time
import pandas as pd


class BCGA:
    def __init__(self, totalFes, domain, component, heatpipe, getObjectiveConstraint, n_p1, n_p2, disc, dism):
        self.totalFes = totalFes
        self.domain = domain
        self.component = component
        self.heatpipe = heatpipe
        self.getObjCons = getObjectiveConstraint

        self.disC = disc
        self.disM = dism
        
        self.follow_count = 0

        self.iniPopu(n_p1, n_p2)

        self.max_shuffle_time = 5
        self.cons_factor = 0.5
        self.fitcount = 0
        self.iteration = 0

        self.iter_best_sol = []
        self.iter_best_val = []

        self.winter_iter = 0
        self.year = self.totalFes // 10
        self.max_scale = np.mean(component.x_opt_max - component.x_opt_min) / 5
        self.proportion = 0.5 # migrant proportion
        self.size_fam = 4

    def iniPopu(self, n_p1, n_p2):
        self.n_p1 = n_p1
        self.n_p2 = n_p2
        self.n_d = self.component.x_opt_dim

        self.popu1_best_objF = None

        rang_l = np.tile(self.component.x_opt_min, (n_p1, 1))
        rang_r = np.tile(self.component.x_opt_max, (n_p1, 1))
        self.popu1 = np.random.uniform(rang_l, rang_r, (self.n_p1, self.n_d))

        rang_l = np.tile(self.component.x_opt_min, (n_p2, 1))
        rang_r = np.tile(self.component.x_opt_max, (n_p2, 1))
        self.popu2 = np.random.uniform(rang_l, rang_r, (self.n_p2, self.n_d))

        self.popu1_best = None

        self.cons1 = np.zeros((n_p1, 4))
        self.objF1 = np.zeros((n_p1, 1))

        self.sol_cache = []
        self.objF_cache = None

        for i in range(n_p1):
            objective, constraint = self.getObjCons(self.popu1[i, :], self.domain, self.component, self.heatpipe)
            obj = objective[0]
            cons1, cons2, cons3, cons4 = constraint
            # conV[i, 0] = sum(constraint)
            self.objF1[i, 0] = obj
            self.cons1[i, :] = constraint


        self.fitcount = self.n_p1

    def GA(self, popu, axis):
        # axis can be 'x' or 'y'

        ParentDec = copy.deepcopy(popu)

        N, D = len(ParentDec), self.n_d

        proC = 0.5
        disC = self.disC
        proM = 2/D
        disM = self.disM
        halfN = len(ParentDec) // 2
        lower = np.tile(self.component.x_opt_min, (N, 1))
        upper = np.tile(self.component.x_opt_max, (N, 1))


        Parent1Dec = ParentDec[0:halfN, :]
        Parent2Dec = ParentDec[halfN:N, :]

        # Simulated binary crossover
        beta = np.zeros((halfN, D))
        mu = np.random.rand(halfN, D)
        beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (disC + 1))
        beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (disC + 1))
        beta = beta * (-1) ** (np.random.randint(2, size=(halfN, D)))
        
        # shut done half
        beta[np.random.rand(halfN, D) > proC] = 1

        # shut done one axis
        if axis == 'x':
            # shut done y
            beta = beta.reshape(-1)
            yindex = np.array([i+j*D for j in range(0,halfN) for i in range(1,D,2)])
            beta[yindex] = 1
            beta = beta.reshape((halfN, D))

        elif axis == 'y':
            # shut done x
            beta = beta.reshape(-1)
            xindex = np.array([i+j*D for j in range(0,halfN) for i in range(0,D,2)])
            beta[xindex] = 1
            beta = beta.reshape((halfN, D))

        else:
            raise ValueError
            
        offSpringDec1 = (Parent1Dec + Parent2Dec) / 2 + beta * (
            Parent1Dec - Parent2Dec
        ) / 2
        offSpringDec2 = (Parent1Dec + Parent2Dec) / 2 - beta * (
            Parent1Dec - Parent2Dec
        ) / 2
        offSpringDec = np.vstack((offSpringDec1, offSpringDec2))
        # if axis == 'x':
        #     pd.DataFrame(ParentDec).to_csv('bug1.csv')
        #     pd.DataFrame(offSpringDec).to_csv('bug2.csv')
        #     exit()
            
        
        # if axis == 'x':
        #     for i in range(N):
        #         for dim in range(0, D, 2):
        #             if beta[i % halfN, dim] == 1 and np.random.rand() < 0.0:
        #                 r = np.random.randint(N)
        #                 offSpringDec[i, dim] = popu[r, dim]

        # else:
            # for dim in range(1, D, 2):
            #     if beta[i % halfN, dim] == 1 and np.random.rand() < 0.1:
            #         r = np.random.randint(N)
            #         offSpringDec[i, dim] = popu[r, dim]

        if axis == 'x':
            p1 = self.p1
        else:
            p1 = 0.8
            disM = 20
        # Polynomial mutation
        if np.random.rand() < p1:

            site1 = np.random.rand(N, D) < proM
            if axis == 'x':
                site1 = site1.reshape(-1)
                yindex = np.array([i+j*D for j in range(0,N) for i in range(1,D,2)])
                site1[yindex] = False
                site1 = site1.reshape(N, D)
            else:
                site1 = site1.reshape(-1)
                xindex = np.array([i+j*D for j in range(0,N) for i in range(0,D,2)])
                site1[xindex] = False
                site1 = site1.reshape(N, D)
            mu = np.random.rand(N, D)
            temp = site1 & (mu <= 0.5)
            offSpringDec[temp] = offSpringDec[temp] + (upper[temp] - lower[temp]) * (
                (
                    2 * mu[temp]
                    + (1 - 2 * mu[temp])
                    * (1 - (offSpringDec[temp] - lower[temp]) / (upper[temp] - lower[temp]))
                    ** (disM + 1)
                )
                ** (1 / (disM + 1))
                - 1
            )
            temp = site1 & (mu > 0.5)
            offSpringDec[temp] = offSpringDec[temp] + (upper[temp] - lower[temp]) * (
                1 - (
                    2 * (1 - mu[temp])
                    + 2 * (mu[temp] - 0.5)
                    * (1 - (upper[temp] - offSpringDec[temp]) / (upper[temp] - lower[temp]))
                    ** (disM + 1)
                )
                ** (1 / (disM + 1))
            )
        
        # dim switch mutation
        else: 
            for id_member in range(len(offSpringDec)):
                if axis == 'x':
                    r_times = np.random.randint(0, self.max_switch_time)
                else:
                    r_times = np.random.randint(0, 2)
                for _ in range(r_times):
                    rdim3, rdim4 = np.random.choice(D // 2, 2, replace = False) # position switch                   
                    if axis == 'x':
                        offSpringDec[id_member, rdim3*2], offSpringDec[id_member, rdim4*2] = offSpringDec[id_member, rdim4*2], offSpringDec[id_member, rdim3*2]
                    else:
                        offSpringDec[id_member, rdim3*2+1], offSpringDec[id_member, rdim4*2+1] = offSpringDec[id_member, rdim4*2+1], offSpringDec[id_member, rdim3*2+1]

        offSpringDec = np.maximum(np.minimum(offSpringDec, upper), lower)

        return offSpringDec

    def normalize_cons(self, cons):
        return cons / (np.max(cons, axis = 0) + 10e-9)
    
    def sum_violation(self, nor_cons):
        cons_count = np.count_nonzero(nor_cons, axis = 1).reshape(len(nor_cons),1)
        ret = (np.sum(nor_cons,axis = 1).reshape(len(nor_cons),1)+cons_count * self.cons_factor).reshape(len(nor_cons),1)
        return ret

    def cal_cost1(self, objF, cons, popu):
        max_objF = np.max(objF, axis = 0)
        nor_cons = self.normalize_cons(cons[ :, 3]).reshape(-1, 1)
        assert nor_cons.shape == (len(objF),1)
        sum_vio = np.count_nonzero(nor_cons, axis = 1).reshape(-1,1)*self.cons_factor + nor_cons
        nor_objF = objF / max_objF * self.cons_factor
        cost = nor_objF * (1-np.count_nonzero(sum_vio, axis = 1)).reshape(len(objF), 1) + sum_vio
        
        pre = self.popu1_best_objF

        feasiIndex = np.where(sum_vio.reshape(-1) == 0)[0]
        if len(feasiIndex) > 0:
            self.popu1_best_objF = np.min(objF[feasiIndex])
        else:
            self.popu1_best_objF = np.Inf
        # self.popu1_best_objF = (np.min(cost)*max_objF/self.cons_factor)[0]
        if self.popu1_best_objF and pre and  round(pre,2) > round(self.popu1_best_objF,2):
            self.need_follow = True
        else:
            self.need_follow = False

        print('best popu1 value :',self.popu1_best_objF, end = ' ')

        best_index = np.argmin(cost)
        self.popu1_best = popu[best_index]

        return cost

    def cal_cost2(self, objF, cons, popu): # for popu2
        max_objF = np.max(objF, axis = 0)
        nor_cons = self.normalize_cons(cons)
        sum_vio = self.sum_violation(nor_cons)
        return sum_vio

    def selection_operator2(self, can_cost):  
        index = np.arange(len(can_cost))
        np.random.shuffle(index)
        shuffle_cost = can_cost[index].reshape(-1, 2)
        winner = np.argmin(shuffle_cost, axis = 1)
        return index[np.arange(len(shuffle_cost)) * 2 + winner]

    def selection_operator1(self, can_cost):
        # family: [[0+i, n_p/2+i, n_p+i, 2n_p/3+i] for i in range(n_p/2)] 
        n_fam = len(can_cost)//self.size_fam
        ret = np.zeros((len(can_cost)//2,), int)
        for i in range(n_fam):
            family_index = np.array([i+j*n_fam for j in range(self.size_fam)])
            family_cost = can_cost[family_index].reshape(-1)
            best_members = self.selection_operator2(family_cost)
            # best_members = np.argsort(family_cost)[:self.size_fam//2]
            best_index = family_index[best_members]
            # i+j*n_fam for j in range(size_fam//2)
            for j in range(self.size_fam//2):
                ret[j*n_fam+i] = best_index[j]
        return ret

    def follow(self):
        for i in range(self.n_p2):
            for dim in range(0, self.n_d, 2):
                self.popu2[i, dim] = self.popu1_best[dim]

    def record_solution(self, popu, cons, objF):
        '''
        Args:
            popu: n_p x n_d
            cons: n_p x 4
            objF: n_p x 1
        '''
        if self.fitcount > self.totalFes*4/10 and np.any(np.sum(cons[:,2:], axis = 1) == 0 ) and self.run_popu2 == False:
            self.run_popu2 = True
            self.ini_popu2 = True
            print('---------run popu2!------------')

        conV = np.sum(cons, axis = 1)
        # record the best solution
        feasiIndex = np.where(conV.reshape(-1) == 0)[0]
        self.feasible_count = len(feasiIndex)
        if len(feasiIndex) > 0:
            # sort the feasible solutions according to their objective
            # function values
            sortedObjF = np.sort(objF.reshape(-1)[feasiIndex])
            index_sortedObjF = np.argsort(objF.reshape(-1)[feasiIndex])
            # the best solution is the feasible solution with minimum 
            # objective function value
            bestvalue = sortedObjF[0]
            bestSolution = popu[feasiIndex[index_sortedObjF[0]], :]
            best_conv = 0
            best_index = feasiIndex[index_sortedObjF[0]]
        else:
            # if the population is infeasbile, the best solution is assigned
            # the value of Inf
            bestvalue = np.Inf
            index_conV = np.argsort(conV.reshape(-1))
            bestSolution = popu[index_conV[0], :]
            best_conv = conV[index_conV[0]]
            best_index = index_conV[0]
        
        return bestvalue, bestSolution, best_conv, best_index

    def main_loop(self):
        self.run_popu2 = False
        self.ini_popu2 = False

        # iters_to_gen_feasible = []

        while self.fitcount < self.totalFes:
            self.p1 = 0.8 - (self.fitcount/self.totalFes)*0.4
            self.max_switch_time = 2 + (self.fitcount/self.totalFes)**2 * 4 
            print(self.p1)
            # print(self.popu1)
            size = [4,8,16,32,64]
            weight = [1,2,4,8,16]
            # pre = self.size_fam
            threshold = [self.totalFes*sum(weight[:i+1])/sum(weight) for i in range(len(size))]
            for i, num in enumerate(threshold):
                if self.fitcount <= num:
                    self.size_fam = size[i]
                    break
            print(self.size_fam)

            # self.size_fam = a[self.fitcount // (1000*self.n_d)]
            # if self.size_fam != pre:
            #     shuffle_index = np.arange(self.n_p1)
            #     np.random.shuffle(shuffle_index)
            #     self.popu1 = self.popu1[shuffle_index]
            self.iteration += 1
        
            print('iteration: ',self.iteration, self.fitcount,'/', self.totalFes)
            new_popu1 = self.GA(self.popu1, axis = 'x')
            new_objF1 = np.zeros((self.n_p1, 1))
            new_cons1 = np.zeros((self.n_p1, 4))
            for i in range(self.n_p1):
                objective, constraint = self.getObjCons(new_popu1[i, :], self.domain, self.component, self.heatpipe)
                obj = objective[0]
                cons1, cons2, cons3, cons4 = constraint
                new_objF1[i, 0] = obj
                new_cons1[i, :] = constraint
                self.fitcount += 1
            
            candidates = np.concatenate((self.popu1, new_popu1), axis = 0)
            can_objF = np.concatenate((self.objF1, new_objF1), axis = 0)
            can_cons = np.concatenate((self.cons1, new_cons1), axis = 0)
            can_cost = self.cal_cost1(can_objF, can_cons, candidates)
            assert(can_cost.shape == (2*self.n_p1, 1))
            selection_index = self.selection_operator1(can_cost)
            self.popu1 = candidates[selection_index]
            self.objF1 = can_objF[selection_index]
            # print(self.objF1)
            self.cons1 = can_cons[selection_index]
            
            self.cur_best_val1, self.cur_best_sol1, self.cur_best_conv1, self.cur_best_index1 = self.record_solution(self.popu1, self.cons1, self.objF1)

            # std.append([np.var(self.objF1)/np.mean(self.objF1),self.popu1_best_objF])

            if not self.run_popu2:
                self.iter_best_val.append(self.cur_best_val1)
                self.iter_best_sol.append(self.cur_best_sol1)
                print()
                # print('iter_bestval: ', [self.cur_best_val1] )
            
            if self.run_popu2:

                if self.ini_popu2:
                    print('---------ini_popu2---------------')
                    self.cur_best_sol2 = None
                    self.objF2 = np.zeros((self.n_p2, 1))
                    self.cons2 = np.zeros((self.n_p2, 4))
                    self.ini_popu2 = False
                    # self.popu2 = self.BF(self.popu2)
                    self.follow()
                    # print(self.popu2)
                    self.popu2 = self.GA(self.popu2, axis = 'y')

                    for i in range(self.n_p2):
                        objective, constraint = self.getObjCons(self.popu2[i, :], self.domain, self.component, self.heatpipe)
                        obj = objective[0]
                        if round(obj, 2) != round(self.popu1_best_objF, 2):
                            raise AssertionError
                        cons1, cons2, cons3, cons4 = constraint
                        self.objF2[i, 0] = obj
                        self.cons2[i, :] = constraint
                        self.fitcount += 1
                    self.cur_best_val2, self.cur_best_sol2, self.cur_best_conv2, self.cur_best_index2 = self.record_solution(self.popu2, self.cons2, self.objF2)

                    print('iter_bestval: ', [self.cur_best_val2])
                    # print('bestconv: ', self.cur_best_conv2)
                elif round(self.cur_best_val2,2) <= round(self.popu1_best_objF,2):
                    if self.objF_cache != self.cur_best_val2:
                        self.sol_cache.append(self.cur_best_sol2)
                        self.objF_cache = self.cur_best_val2
                        # iters_to_gen_feasible.append(self.follow_count)
                        # print(iters_to_gen_feasible)
                    print(self.objF_cache, 'feasible')

                else:
                    # print(self.popu2)
                    if self.need_follow:
                        self.follow()
                        for i in range(self.n_p2):
                            objective, constraint = self.getObjCons(self.popu2[i, :], self.domain, self.component, self.heatpipe)
                            obj = objective[0]
                            if round(obj, 2) != round(self.popu1_best_objF, 2):
                                raise AssertionError
                            cons1, cons2, cons3, cons4 = constraint
                            self.objF2[i, 0] = obj
                            self.cons2[i, :] = constraint
                            self.fitcount += 1
                        # self.follow_count = 0
                    # new_popu2 = self.BF(self.popu2)
                    # self.follow_count+=1
                    new_popu2 = self.GA(self.popu2, axis = 'y')
                    new_objF2 = np.zeros((self.n_p2, 1))
                    new_cons2 = np.zeros((self.n_p2, 4))
                    for i in range(self.n_p2):
                        objective, constraint = self.getObjCons(new_popu2[i, :], self.domain, self.component, self.heatpipe)
                        obj = objective[0]
                        if round(obj, 2) != round(self.popu1_best_objF,2):
                            raise AssertionError

                        cons1, cons2, cons3, cons4 = constraint
                        new_objF2[i, 0] = obj
                        new_cons2[i, :] = constraint
                        self.fitcount += 1

                    candidates = np.concatenate((self.popu2, new_popu2), axis = 0)
                    can_objF = np.concatenate((self.objF2, new_objF2), axis = 0)
                    can_cons = np.concatenate((self.cons2, new_cons2), axis = 0)
                    can_cost = self.cal_cost2(can_objF, can_cons, candidates)
                    assert(can_cost.shape == (2*self.n_p2, 1))
                    selection_index = self.selection_operator2(can_cost)
                    self.popu2 = candidates[selection_index]
                    self.objF2 = can_objF[selection_index]
                    self.cons2 = can_cons[selection_index]

                    self.cur_best_val2, self.cur_best_sol2, self.cur_best_conv2, self.cur_best_index2 = self.record_solution(self.popu2, self.cons2, self.objF2)
                    print('iter_bestval: ', [self.cur_best_val2])
                    # print('bestconv: ', self.cur_best_conv2)

                if self.objF_cache:
                    self.iter_best_val.append(self.objF_cache)
                    self.iter_best_sol.append(self.sol_cache[-1])
                else:
                    self.iter_best_val.append(self.cur_best_val2)
                    self.iter_best_sol.append(self.cur_best_sol2)
                print(self.iter_best_val[-1])
        # pd.DataFrame(std).to_csv(f'{self.totalFes}.csv')
        # pd.DataFrame(iters_to_gen_feasible).to_csv('iters_to_gen_feasi.csv')
        return self.sol_cache[-1], self.iter_best_sol, self.iter_best_val

        

def main_algorithm(totalFes, domain, component, heatpipe, getObjectiveConstraint):
    bpa = BCGA(
        totalFes,
        domain,
        component,
        heatpipe, 
        getObjectiveConstraint,
        n_p1 = 32,
        n_p2 = 60,
        disc = 20,
        dism = 5
        )
    return bpa.main_loop()


