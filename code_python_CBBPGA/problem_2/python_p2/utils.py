# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2021/12/09 22:30:06
@Author  :   Xianqi
@Contact :   chenxianqi12@qq.com
'''


import pickle
import numpy as np
import matplotlib.pyplot as plt


def save_result(data, filename="result.csv"):
    """保存结果到csv文件"""
    np.savetxt(filename, data, delimiter=",")


def read_result(filename="result.csv"):
    """读取csv结果"""
    x = np.loadtxt(filename, delimiter=",")
    return x.reshape(1, -1)


def save_variable(filename, data):
    """保存变量到文件中"""
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_variable(filename):
    """从文件读取变量"""
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def overlap_rec_rec(u1, a1, b1, u2, a2, b2):
    """
    可同时处理多组矩形组件之间的干涉计算。
    :param : u1, u2 两组件中心点坐标 n*2
             a1, b1 组件1 长、宽的一半 n*1
             a2, b2 组件2 长、宽的一半 n*1
    :return : overlap_area 干涉面积 n*1
    """
    Phi1 = np.minimum(
        np.abs(u1[:, 0].reshape([-1, 1]) - u2[:, 0].reshape([-1, 1]))
        - a1.reshape([-1, 1])
        - a2.reshape([-1, 1]),
        0,
    )
    Phi2 = np.minimum(
        np.abs(u1[:, 1].reshape([-1, 1]) - u2[:, 1].reshape([-1, 1]))
        - b1.reshape([-1, 1])
        - b2.reshape([-1, 1]),
        0,
    )
    overlap_area = (-Phi1) * (-Phi2)
    return overlap_area


def distance_rec_rec(u1, a1, b1, u2, a2, b2):
    """
    可同时处理多组矩形组件之间的近似距离计算。
    :param : u1, u2 两组件中心点坐标 n*2
             a1, b1 组件1 长、宽的一半 n*1
             a2, b2 组件2 长、宽的一半 n*1
    :return : distance 组件之间的距离 n*1
    """
    Phi1 = (
        np.abs(u1[:, 0].reshape([-1, 1]) - u2[:, 0].reshape([-1, 1]))
        - a1.reshape([-1, 1])
        - a2.reshape([-1, 1])
    )
    Phi2 = (
        np.abs(u1[:, 1].reshape([-1, 1]) - u2[:, 1].reshape([-1, 1]))
        - b1.reshape([-1, 1])
        - b2.reshape([-1, 1])
    )

    distance = np.maximum(Phi1, Phi2)
    return distance


def overlap_components(comps1_location, comps1_size, *args):
    """
    计算组件系统comps1和comps2之间的相互距离

    Args:
        comps1_location 组件系统1的位置坐标 N*2
        comps1_size 组件系统1的尺寸坐标 N*2
        [*args]: 当为空时，表示计算comps1系统内组件与组件之间的距离
                 当不为空时，应输入(comps2_location, comps2_size)

    Returns:
        overlap 干涉面积 (N * N) 或者 (N * M)
    """
    if len(args) == 0:
        comps2_location = comps1_location
        comps2_size = comps1_size
    elif len(args) == 2:
        comps2_location = args[0]
        comps2_size = args[1]
    else:
        raise ValueError("Please input two or four parameters.")

    comps1_num = comps1_location.shape[0]
    comps2_num = comps2_location.shape[0]

    overlap = np.zeros((comps1_num, comps2_num))
    for ind1 in range(comps1_num):
        if len(args) == 0:
            ind2 = ind1 + 1
        else:
            ind2 = 0
        length = comps2_num - ind2

        u1 = comps1_location[ind1, :].reshape(1, -1).repeat(length, axis=0)
        u2 = comps2_location[ind2::, :].reshape(-1, 2)

        a1 = (comps1_size[ind1, 0] / 2).reshape(-1, 1).repeat(length, axis=0)
        b1 = (comps1_size[ind1, 1] / 2).reshape(-1, 1).repeat(length, axis=0)
        a2 = (comps2_size[ind2::, 0] / 2).reshape(-1, 1)
        b2 = (comps2_size[ind2::, 1] / 2).reshape(-1, 1)

        overlap_volume = overlap_rec_rec(u1, a1, b1, u2, a2, b2)
        overlap[ind1, ind2::] = overlap_volume[:, 0]

    if len(args) < 2:
        overlap = overlap + overlap.T

    return overlap


def distance_components(comps1_location, comps1_size, *args):
    """
    计算组件系统comps1和comps2之间的相互距离

    Args:
        comps1_location 组件系统1的位置坐标 N*2
        comps1_size 组件系统1的尺寸坐标 N*2
        [*args]: 当为空时，表示计算comps1系统内组件与组件之间的距离
                 当不为空时，应输入(comps2_location, comps2_size)

    Returns:
        distance 组件距离 (N * N) 或者 (N * M)
    """
    if len(args) == 0:
        comps2_location = comps1_location
        comps2_size = comps1_size
    elif len(args) == 2:
        comps2_location = args[0]
        comps2_size = args[1]
    else:
        raise ValueError("Please input two or four parameters.")

    comps1_num = comps1_location.shape[0]
    comps2_num = comps2_location.shape[0]

    distance = np.zeros((comps1_num, comps2_num))
    for ind1 in range(comps1_num):
        if len(args) == 0:
            ind2 = ind1 + 1
        else:
            ind2 = 0
        length = comps2_num - ind2

        u1 = comps1_location[ind1, :].reshape(1, -1).repeat(length, axis=0)
        u2 = comps2_location[ind2::, :].reshape(-1, 2)

        a1 = (comps1_size[ind1, 0] / 2).reshape(-1, 1).repeat(length, axis=0)
        b1 = (comps1_size[ind1, 1] / 2).reshape(-1, 1).repeat(length, axis=0)
        a2 = (comps2_size[ind2::, 0] / 2).reshape(-1, 1)
        b2 = (comps2_size[ind2::, 1] / 2).reshape(-1, 1)

        overlap_volume = distance_rec_rec(u1, a1, b1, u2, a2, b2)
        distance[ind1, ind2::] = overlap_volume[:, 0]

    if len(args) < 2:
        distance = distance + distance.T

    return distance


def Fun1_overlap(x, domain, component):
    """
    根据输入组件的位置向量，计算组件与组件之间的干涉

    Args:
        x: position variables (dim: 1x2n)
        domain: Parameters中设置参数
        component: Parameters中设置参数

    Returns:
        total_overlap_volume: 总干涉量
    """
    comp_position = x.reshape(-1, 2)  # 119x2
    overlap_volume = []

    for i in range(domain.part_num):  # 遍历 left - right 
        sat_layout_domain = domain.part[i]
        for j in range(sat_layout_domain.part_num):  # 遍历 down - up
            layout_domain = sat_layout_domain.part[j]
            comp_num = layout_domain.comp_num
            if comp_num == 0:  # 表示该布局区域内没有组件
                continue

            # 准备布局区域尺寸
            dom_sub_location = layout_domain.recDomRep.rec_position
            dom_sub_size = layout_domain.recDomRep.rec_size

            # 准备该区域内组件尺寸
            comp_sub_location = np.zeros((comp_num, 2))
            comp_sub_size = np.zeros((comp_num, 2))
            for ind, name in enumerate(layout_domain.comp_id):
                index = component.name_index[name]
                comp_sub_location[ind, :] = comp_position[index, :]
                comp_sub_size[ind, :] = component.comp_size[index, :]
            
            sub_location = np.vstack((dom_sub_location, comp_sub_location))
            sub_size = np.vstack((dom_sub_size, comp_sub_size))
            sub_overlap = overlap_components(sub_location, sub_size)
            sub_overlap_volume = np.sum(sub_overlap) / 2
            overlap_volume.append(sub_overlap_volume)
    
    total_overlap_volume = np.sum(overlap_volume)
    return total_overlap_volume


def Fun2_systemcentroid(x, component):
    """
    根据输入组件的位置向量，计算系统的质心位置(y方向质心坐标)

    Args:
        x: position variables (dim: 1x2n)
        component: Parameters中设置参数

    Returns:
        yc: y方向质心
    """
    comp_position = x.reshape(-1, 2)  # 119x2
    comp_mass = component.comp_mass
    yc = np.sum(comp_position[:, 1].reshape(-1, 1) * comp_mass) / np.sum(comp_mass)
    return yc


def Fun3_overlap_heatpipe(x, domain, component, heatpipe, safe_dis=0):
    """
    根据输入组件的位置向量，计算组件和热管的干涉情况

    Args:
        x: position variables (dim: 1x2n)
        heatpipe: Parameters中设置参数
        component: Parameters中设置参数
        safe_dis: 组件占据热管的最小宽度

    Returns:
        overlap: (type: list) [overlap_left, overlap_right]
            overlap_left: 左板上干涉情况 (dim: 16x66) (若没有组件则为None)
            overlap_right: 右板上干涉情况 (dim: 16x53) (若没有组件则为None)
    """
    comp_position = x.reshape(-1, 2)  # 119x2

    overlap = []
    for i in range(domain.part_num):  # 遍历 left - right
        sat_layout_domain = domain.part[i]
        if sat_layout_domain.comp_num == 0:
            # 表示该板上没有组件
            overlap.append(None)
            continue
        
        # 载入布局组件信息
        comp_location = []
        comp_size = []
        for name in sat_layout_domain.comp_id:
            index = component.name_index[name]
            comp_location.append(comp_position[index, :].tolist())
            comp_size.append(component.comp_size[index, :].tolist())
        comp_location = np.array(comp_location)
        comp_size = np.array(comp_size) - safe_dis

        # 载入热管信息
        hp_location = []
        hp_size = []
        for name in sat_layout_domain.hp_id:
            index = heatpipe.name_index[name]
            hp_location.append(heatpipe.location[index, :].tolist())
            hp_size.append(heatpipe.hp_size[index, :].tolist())
        hp_location = np.array(hp_location)
        hp_size = np.array(hp_size)
        hp_size[:, 0] = hp_size[:, 0] - safe_dis

        distance_hp_comp = distance_components(hp_location, hp_size, comp_location, comp_size)
        overlap.append(- distance_hp_comp)
    return overlap  # overlap > 0 代表存在干涉


def Fun3_heatpower(x, domain, component, heatpipe):
    """
    根据输入组件的位置向量，计算组件和热管的干涉情况

    Args:
        x: position variable (dim: 1x2n)
        heatpipe: Parameters中设置参数
        component: Parameters中设置参数

    Returns:
        hep_power: [hep_left, hep_right] 每根热管的总承载量 (dim: 2xhep_num)
        comp_hep_dis: 输出不和热管相交的组件 离最近一根热管的距离 (dim: nx1)
    """
    # 调用干涉计算函数 计算热管和组件的干涉情况
    overlap = Fun3_overlap_heatpipe(
        x, domain, component, heatpipe, safe_dis=heatpipe.width * 0.99
    )  # 此处取 0.99 为了确保当刚好边界接触时 能够判断为 干涉
    overlap_dis = Fun3_overlap_heatpipe(
        x, domain, component, heatpipe, safe_dis=heatpipe.width
    )  # 要求完全横跨热管才可以满足散热要求，故组件占据热管的最小距离为热管的宽度

    comp_intensity = component.comp_intensity_backup.copy().reshape(-1, 1)

    hp_power = [0] * domain.part_num
    comp_hep_dis = [0] * domain.part_num
    for i in range(domain.part_num):  # 遍历 left - right
        sat_layout_domain = domain.part[i]
        comp_num = sat_layout_domain.comp_num
        if comp_num == 0:
            # 表示该板上没有组件
            hp_power[i] = np.zeros((sat_layout_domain.hp_num, ))
            comp_hep_dis[i] = np.zeros((1, comp_num))
            continue
        
        comp_intensity_part = []
        for name in sat_layout_domain.comp_id:
            index = component.name_index[name]
            comp_intensity_part.append(comp_intensity[index, 0])
        comp_intensity_part = np.array(comp_intensity_part).reshape(1, -1)
        
        # 根据每个组件实际跨越几根热管 平均分配热耗
        overlap_part = overlap[i]
        overlapflag_part = np.zeros_like(overlap_part)
        overlapflag_part[overlap_part > 0] = 1
        comp_in_hep_num = np.sum(overlapflag_part, axis=0).reshape(1, -1)

        # 计算不满足干涉约束时 返回所离最近热管的距离
        overlap_dis_part = overlap_dis[i]
        comp_hep_dis_part = np.zeros((1, comp_num))
        if np.min(comp_in_hep_num) < 1:  # 说明存在组件没有骑在热管上
            # 返回没有骑在热管上组件 距其最近热管的距离 >0 [dim: 1 x comp_num]
            temp = - np.max(overlap_dis_part, axis=0).reshape(1, -1)
            # logicals 没有骑在热管上组件的编号为1时 代表没有骑在热管上  [dim: 1 x comp_num]
            comp_index_part = comp_in_hep_num < 1
            comp_hep_dis_part[comp_index_part] = temp[comp_index_part]  # [dim: 1 x comp_num]
        
        # 平均分配热功耗
        comp_intensity_part_real = comp_intensity_part.copy()
        comp_intensity_part_real[comp_in_hep_num > 0] = (
            comp_intensity_part_real[comp_in_hep_num > 0] / comp_in_hep_num[comp_in_hep_num > 0]
        )

        # 计算每根热管上的总功耗
        hp_num = sat_layout_domain.hp_num
        hp_power_part_matrix = (comp_intensity_part_real.repeat(hp_num, axis=0)) * overlapflag_part  # hp_num x comp_num
        hp_power_part = np.sum(hp_power_part_matrix, axis=1)  # hp_num x 1
        
        # 保存结果
        hp_power[i] = hp_power_part.reshape(-1)
        comp_hep_dis[i] = comp_hep_dis_part
    
    # 结果形式整理 
    comp_hep_dis = np.hstack((comp_hep_dis[0], comp_hep_dis[1]))  # 1 x comp_num_total

    return (hp_power, comp_hep_dis)


def getObjectiveConstraint(x, domain, component, heatpipe):
    """
    根据输入组件的位置向量，计算目标函数和约束函数

    Args:
        x: position variable (dim: 1x2n)
        domain: Parameters中设置参数
        heatpipe: Parameters中设置参数
        component: Parameters中设置参数

    Returns:
        objective: [list] [obj1, ...]
        constraint: [list] [cons1, cons2, cons3, cons4, ...] (requires: <= 0)
    """
    objective = []
    constraint = []

    hp_power, comp_hep_dis = Fun3_heatpower(x, domain, component, heatpipe)
    
    # obj1: maximum heatpipe dissipation power
    obj1 = 0
    obj1 += sum([np.max(i) for i in hp_power])
    objective.append(obj1)

    # cons1: non-overlapping constraint
    overlap = Fun1_overlap(x, domain, component)
    cons1 = overlap
    constraint.append(cons1)

    # cons2: system centroid constraint
    yc = Fun2_systemcentroid(x, component)
    yc_ex = 250
    delta_yc = 5
    cons2 = max([np.abs(yc - yc_ex) - delta_yc, 0])
    constraint.append(cons2)

    # cons3: maximum heatpipe dissipation power constraint
    hp_maxload = heatpipe.hp_maxload.reshape(-1)
    hp_power_total = np.hstack((hp_power[0], hp_power[1])).reshape(-1)
    cons3 = np.sum(hp_power_total[hp_power_total > hp_maxload] - hp_maxload[hp_power_total > hp_maxload])
    constraint.append(cons3)

    # cons4: overlapping constraint between components and heatpipes
    cons4 = np.sum(comp_hep_dis)
    constraint.append(cons4)

    # objective and constraint
    return objective, constraint


def Interpreter(x_opt, component):
    """将优化变量x_opt转化为所有组件对应的位置向量x_com

    Args:
        x_opt: 1x2m np.ndarray
        component: class-Component

    Returns:
        x_com: 1x2n [x1, y1, ..., xn, yn]
    """
    x_com = component.interpreter_x_opt2pos(x_opt)
    return x_com


def getCompOccupyHp_CandidateNum(comp_width, hp_width, hp_interval, safe_dis):
    """
    根据组件的宽度计算出组件允许横跨热管的数量

    Args:
        comp_width: 组件宽度 (dim: n)
        hp_width: 热管宽度 (dim: 1)
        hp_interval: 热管间距  (dim: 1)
        safe_dis: 组件横跨热管需占据的最小宽度 (dim: 1)
    
    Returns:
        candidateNum: [list] (dim: n) [[1, 2], [2, 3], [5, 6], ...]
    """
    if isinstance(comp_width, list):
        n = len(comp_width)
    elif isinstance(comp_width, np.ndarray):
        n = comp_width.size
    elif isinstance(comp_width, (float, int)):
        n = 1
    candidateNum = [0] * n

    for i in range(n):
        if n == 1:
            cp_width = comp_width
        else:
            cp_width = comp_width[i]
        candNum = []

        width1_max = hp_interval + 2 * safe_dis  # 占据一根热管的最大长度
        if cp_width <= 0:
            raise ValueError(f"第 {i+1} 个组件宽度应该大于0.")
        elif cp_width < width1_max:
            candNum = [1]
            candidateNum[i] = candNum
        else:
            candNum_float = (cp_width + 2 * hp_width + hp_interval - 2 * safe_dis) / (hp_width + hp_interval)
            max_candNum = np.floor(candNum_float).astype(int)
            candNum = [max_candNum - 1, max_candNum]
            candidateNum[i] = candNum
    return candidateNum


def plot_layout(x, domain, component, heatpipe, prefix_name="temp", savefig=False, disfig=True, dismass=False):
    """画出布局图

    Args:
        x: position variables (dim: 1x2n)
    """
    comp_position = x.reshape(-1, 2)

    for i in range(domain.part_num):  # 遍历南北板 left - right
        sat_layout_domain = domain.part[i]
        if sat_layout_domain.comp_num == 0:
            # 当前板上如果没有组件 则不作图
            continue
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # 画出上下两个布局区域
        flag = 1
        for j in range(sat_layout_domain.part_num):  # 遍历 down - up
            layout_domain = sat_layout_domain.part[j]
            if layout_domain.comp_num == 0:
                # 当前布局区域如果没有组件 则不作图
                flag = 0
                continue
            
            # 画出当前布局区域边界
            width = layout_domain.width
            height = layout_domain.height
            dom_pos = (layout_domain.location[0, 0], layout_domain.location[0, 1])
            rec_dom = plt.Rectangle(dom_pos, width, height, fill=False, color="red")
            ax.add_patch(rec_dom)

            # 画出当前布局区域内所有组件
            for ind, name in enumerate(layout_domain.comp_id):
                index = component.name_index[name]
                comp_location = comp_position[index, :]
                comp_size = component.comp_size[index, :]
                comp_leftdown_location = comp_location - comp_size / 2
                rect_comp = plt.Rectangle(
                    comp_leftdown_location,
                    comp_size[0],
                    comp_size[1],
                    fill=False,
                    color="blue",
                    linewidth=0.5,
                )
                ax.add_patch(rect_comp)
                if dismass:
                    plt.text(
                        comp_leftdown_location[0],
                        comp_leftdown_location[1] + comp_size[1] / 2,
                        f"{index+1}, {component.comp_mass[index, 0]}kg",
                        fontsize=6,
                        va='center',
                    )
                else:
                    plt.text(
                        comp_leftdown_location[0],
                        comp_leftdown_location[1] + comp_size[1] / 2,
                        f"{index+1}",
                        fontsize=6,
                        va='center',
                    )
            
            # 画出当前布局区域内热管
            hp_num = layout_domain.hp_num
            for ind, name in enumerate(layout_domain.hp_id):
                index = heatpipe.name_index[name]
                hp_location = heatpipe.location[index, :]
                hp_size = heatpipe.hp_size[index, :]
                hp_leftdown_location = hp_location - hp_size / 2
                rect_hep = plt.Rectangle(
                    hp_leftdown_location,
                    hp_size[0],
                    hp_size[1],
                    fill=False,
                    color="lime",
                    linewidth=0.5,
                    linestyle="dashed",
                )
                ax.add_patch(rect_hep)
            
            axis_expand_ratio = 0.01
            x_lb = layout_domain.location[0, 0] - axis_expand_ratio * width
            x_ub = layout_domain.location[0, 2] + axis_expand_ratio * width
            y_lb = layout_domain.location[0, 1] - axis_expand_ratio * height
            y_ub = layout_domain.location[0, 3] + axis_expand_ratio * height
            plt.axis([x_lb, x_ub, y_lb, y_ub])
            # plt.text((x_lb + x_ub) / 2, (y_lb + y_ub) / 2, layout_domain.name, fontsize=15, fontstyle='italic', alpha=0.2, va='center')
        
        if flag > 0:
            width = sat_layout_domain.width
            height = sat_layout_domain.height
            dom_pos = (sat_layout_domain.location[0, 0], sat_layout_domain.location[0, 1])
            rec_dom = plt.Rectangle(dom_pos, width, height, fill=False, color="red", linewidth=1, linestyle="dashed")
            ax.add_patch(rec_dom)

            x_lb = sat_layout_domain.location[0, 0] - axis_expand_ratio * width
            x_ub = sat_layout_domain.location[0, 2] + axis_expand_ratio * width
            y_lb = sat_layout_domain.location[0, 1] - axis_expand_ratio * height
            y_ub = sat_layout_domain.location[0, 3] + axis_expand_ratio * height
            plt.axis([x_lb, x_ub, y_lb, y_ub])
            plt.text((x_lb + x_ub) / 2, (y_lb + y_ub) / 2, sat_layout_domain.name.upper(), family='sans-serif', va='center')
        
        if savefig:
            prefix = str(prefix_name)
            left_name = prefix + "_" + sat_layout_domain.name + ".jpg"  # "Layout_left_opt3.jpg"
            plt.savefig(left_name, dpi=300, bbox_inches="tight")
        if disfig:
            plt.show()
        plt.close()
    return None


def prepare_data(json_file):
    import json
    from Param import JsonDecode
    
    # json_file = "JsonData-2021-12-09_23_49_05.json"
    f = open(json_file, encoding="utf-8")
    json_data = json.load(f)

    jd = JsonDecode(json_data)
    domain = jd.decode_domain()
    component = jd.decode_component()
    heatpipe = jd.decode_heatpipe()
    return domain, component, heatpipe


def test_Fun1_overlap():
    json_files = test_json_files()
    domain, component, heatpipe = prepare_data(json_files[0])

    # test: Fun1_overlap()
    # (1) 读取当前组件位置坐标
    x = component.location.reshape(1, -1)

    # (2) 随机产生一组组件位置坐标
    # x_max, x_min = component.location_max, component.location_min
    # x = x_min + np.random.rand(x_min.shape[0], x_min.shape[1]) * (x_max - x_min)

    overlap = Fun1_overlap(x, domain, component)
    print(f"Overlap volume = {overlap}")

    plot_layout(x, domain, component, heatpipe, prefix_name="test", savefig=True, disfig=False, dismass=False)
    return None


def test_Fun2_systemcentroid():
    json_files = test_json_files()
    domain, component, heatpipe = prepare_data(json_files[0])

    # (1) 读取当前组件位置坐标
    x = component.location.reshape(1, -1)

    # (2) 随机产生一组组件位置坐标
    # x_max, x_min = component.location_max, component.location_min
    # x = x_min + np.random.rand(x_min.shape[0], x_min.shape[1]) * (x_max - x_min)

    yc = Fun2_systemcentroid(x, component)
    print(f"System centroid (y): {yc:,.2f}")


def test_Fun3_heatpower():
    json_files = test_json_files()
    domain, component, heatpipe = prepare_data(json_files[0])

    # (1) 读取当前组件位置坐标
    x = component.location.reshape(1, -1)

    # (2) 随机产生一组组件位置坐标
    # x_max, x_min = component.location_max, component.location_min
    # x = x_min + np.random.rand(x_min.shape[0], x_min.shape[1]) * (x_max - x_min)

    hp_power, comp_hep_dis = Fun3_heatpower(x, domain, component, heatpipe)
    print(f"Components number: {domain.part[0].comp_num}, {domain.part[1].comp_num}")
    print(f"Heatpipes loading ({domain.part[0].name}, {np.sum(hp_power[0])} W): {hp_power[0].tolist()} W")
    print(f"Heatpipes loading ({domain.part[1].name}, {np.sum(hp_power[1])} W): {hp_power[1].tolist()} W")
    print(f"Constraint violation: {comp_hep_dis.reshape(-1).tolist()}")
    print(f"----------------------------------------------------")


def test_getObjectiveConstraint():
    json_files = test_json_files()
    domain, component, heatpipe = prepare_data(json_files[0])

    # (1) 读取当前组件位置坐标
    x = component.location.reshape(1, -1)

    # (2) 随机产生一组组件位置坐标
    # x_max, x_min = component.location_max, component.location_min
    # x = x_min + np.random.rand(x_min.shape[0], x_min.shape[1]) * (x_max - x_min)

    # (3) 根据优化变量范围 随机 产生一组组件位置坐标
    # x_max, x_min = component.x_opt_max, component.x_opt_min
    # x_opt = x_min + np.random.rand(x_min.shape[0], x_min.shape[1]) * (x_max - x_min)
    # x = Interpreter(x_opt, component)

    obj, cons = getObjectiveConstraint(x, domain, component, heatpipe)
    obj_num = len(obj)
    cons_num = len(cons)

    print(f"Component number: {component.number}")
    print(f"Component position: ")
    print(f"x 坐标: {x.reshape(-1, 2).T[0, :].tolist()}")
    print(f"y 坐标: {x.reshape(-1, 2).T[1, :].tolist()}")
    print(f"Objectives ({obj_num}): {obj}")
    print(f"Constraints ({cons_num}): {cons}")
    
    plot_layout(x, domain, component, heatpipe, prefix_name="test1", savefig=True, disfig=False, dismass=True)
    print(f"----------------------------------------------------")

    return None


def test_Interpreter():
    json_files = test_json_files()
    domain, component, heatpipe = prepare_data(json_files[0])

    # (1) 读取当前组件位置坐标
    x = component.location.reshape(1, -1)

    print(f"Component position: ")
    print(f"{x.reshape(-1).tolist()}")

    # (2) 优化变量 x_opt
    x_opt = component.interpreter_x_pos2opt(x)
    print(f"Optimization variables: {component.x_opt_dim}")
    print(f"{x_opt.reshape(-1).tolist()}")

    # (3) 还原组件位置 x_com
    x_com = component.interpreter_x_opt2pos(x_opt)
    print(f"Returned Optimization variables: ")
    print(f"{x_com.reshape(-1).tolist()}")
    
    # 打出固定位置组件编号, 备份组件编号
    locked_index = (np.where(component.lock_indicator[:, 0] > 0))[0] + 1
    print(f"Locked components: {locked_index.tolist()} (Begin from 1)")

    backup_index = []
    for group in component.groups.keys():
        comps = component.groups[group]['comps']
        comps_index = []
        for name in comps:
            comps_index.append(component.name_index[name] + 1)
        backup_index.append(comps_index)
    print(f"Backup components: {backup_index} (Begin from 1)")
    print(f"----------------------------------------------------")


def test_Plot_layout():
    json_files = test_json_files()
    domain, component, heatpipe = prepare_data(json_files[1])

    x = component.location.reshape(1, -1)

    plot_layout(x, domain, component, heatpipe, prefix_name="test1", savefig=False, disfig=True, dismass=True)


def test_getCompOccupyHp_CandidateNum():
    comp_width = 479
    # comp_width = [30, 150, 180, 479, 480]
    # comp_width = np.array([30, 150, 180, 479, 480])
    hp_width = 30
    hp_interval = 120
    safe_dis = 30
    candNum = getCompOccupyHp_CandidateNum(comp_width, hp_width, hp_interval, safe_dis)
    print(f"Component width: {comp_width}")
    print(f"Possible occupied heatpipe nums: {candNum}")


def test_json_files():
    json_files = []
    # json_files.append("JsonData-2021-12-09_23_49_05.json")
    # json_files.append("JsonData-2021-12-11_16_52_38.json")
    # json_files.append("JsonData-2021-12-11_17_16_22.json")
    # json_files.append("JsonData-2021-12-12_00_01_43.json")  # 测试备份和位置锁定
    # json_files.append("JsonData-2021-12-16_14_36_59.json")
    json_files.append("Problem2_15comps_1domain.json")

    for i, json_file in enumerate(json_files):
        json_files[i] = "json_files/" + json_file
    return json_files


if __name__ == "__main__":
    # 1. test Fun1_overlap
    # test_Fun1_overlap()

    # 2. test Fun2_systemcentroid
    # test_Fun2_systemcentroid()

    # 3. test Fun3_heatpower
    test_Fun3_heatpower()

    # 4. test getObjectiveConstraint
    test_getObjectiveConstraint()

    # 5. test Interpreter
    test_Interpreter()

    # 6. test Plot_layout
    # test_Plot_layout()

    # 7. test getCompOccupyHp_CandidateNum
    # test_getCompOccupyHp_CandidateNum()
