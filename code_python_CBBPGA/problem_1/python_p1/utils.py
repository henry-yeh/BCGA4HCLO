import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle


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
        x: position variable (dim: 1x2n)
        domain: Parameters中设置参数
        component: Parameters中设置参数

    Returns:
        overlap: 总干涉量
    """
    comp_position = x.reshape(-1, 2)  # 119x2

    # 左板下 plane = 1
    num1 = component.comp_num_plane[0]
    dom_sub1_location = domain.subdomain1.rec_position
    dom_sub1_size = domain.subdomain1.rec_size
    com_sub1_location = comp_position[0:num1, :]
    com_sub1_size = component.comp_size[0:num1, :]
    sub1_location = np.vstack((dom_sub1_location, com_sub1_location))
    sub1_size = np.vstack((dom_sub1_size, com_sub1_size))
    overlap1 = overlap_components(sub1_location, sub1_size)
    overlap_volume1 = np.sum(overlap1) / 2

    # 左板上
    num2 = np.sum(component.comp_num_plane[0:2])
    dom_sub2_location = domain.subdomain2.rec_position
    dom_sub2_size = domain.subdomain2.rec_size
    com_sub2_location = comp_position[num1:num2, :]
    com_sub2_size = component.comp_size[num1:num2, :]
    sub2_location = np.vstack((dom_sub2_location, com_sub2_location))
    sub2_size = np.vstack((dom_sub2_size, com_sub2_size))
    overlap2 = overlap_components(sub2_location, sub2_size)
    overlap_volume2 = np.sum(overlap2) / 2

    # 右板下
    num3 = np.sum(component.comp_num_plane[0:3])
    dom_sub3_location = domain.subdomain3.rec_position
    dom_sub3_size = domain.subdomain3.rec_size
    com_sub3_location = comp_position[num2:num3, :]
    com_sub3_size = component.comp_size[num2:num3, :]
    sub3_location = np.vstack((dom_sub3_location, com_sub3_location))
    sub3_size = np.vstack((dom_sub3_size, com_sub3_size))
    overlap3 = overlap_components(sub3_location, sub3_size)
    overlap_volume3 = np.sum(overlap3) / 2

    # 右板上
    # num4 = np.sum(component.comp_num_plane)
    dom_sub4_location = domain.subdomain4.rec_position
    dom_sub4_size = domain.subdomain4.rec_size
    com_sub4_location = comp_position[num3::, :]
    com_sub4_size = component.comp_size[num3::, :]
    sub4_location = np.vstack((dom_sub4_location, com_sub4_location))
    sub4_size = np.vstack((dom_sub4_size, com_sub4_size))
    overlap4 = overlap_components(sub4_location, sub4_size)
    overlap_volume4 = np.sum(overlap4) / 2

    # 总干涉量
    overlap_volume = (
        overlap_volume1 + overlap_volume2 + overlap_volume3 + overlap_volume4
    )
    return overlap_volume


def Fun2_systemcentroid(x, component):
    """
    根据输入组件的位置向量，计算系统的质心位置(y方向质心坐标)

    Args:
        x: position variable (dim: 1x2n)
        component: Parameters中设置参数

    Returns:
        yc: y方向质心
    """
    comp_position = x.reshape(-1, 2)  # 119x2
    comp_mass = component.mass
    yc = np.sum(comp_position[:, 1].reshape(-1, 1) * comp_mass) / np.sum(comp_mass)
    return yc


def Fun3_overlap_heatpipe(x, heatpipe, component, safe_dis=0):
    """
    根据输入组件的位置向量，计算组件和热管的干涉情况

    Args:
        x: position variable (dim: 1x2n)
        heatpipe: Parameters中设置参数
        component: Parameters中设置参数

    Returns:
        overlap_left: 左板上干涉情况 (dim: 16x66)
        overlap_right: 右板上干涉情况 (dim: 16x53)
    """
    comp_position = x.reshape(-1, 2)  # 119x2
    comp_size = copy.deepcopy(component.comp_size) - safe_dis

    # 对热功耗进行计算时应该将一个板上的一起考虑
    # 左板上下一起计算
    num1 = np.sum(component.comp_num_plane[0:2])
    hep_num = heatpipe.number
    hep_location = heatpipe.position
    hep_size = heatpipe.rec_size.reshape(1, -1).repeat(hep_num, axis=0) - safe_dis
    comp_left_location = comp_position[0:num1, :]
    comp_left_size = comp_size[0:num1, :]
    distance_left = distance_components(
        hep_location, hep_size, comp_left_location, comp_left_size
    )
    overlap_left = -distance_left

    # 右板上下一起计算
    # num2 = np.sum(component.comp_num_plane)
    comp_right_location = comp_position[num1::, :]
    comp_right_size = comp_size[num1::, :]
    distance_right = distance_components(
        hep_location, hep_size, comp_right_location, comp_right_size
    )
    overlap_right = -distance_right
    return (overlap_left, overlap_right)  # overlap > 0 代表存在干涉


def Fun3_heatpower(x, heatpipe, component):
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
    overlap_left, overlap_right = Fun3_overlap_heatpipe(
        x, heatpipe, component, safe_dis=0.99*heatpipe.width
    )
    overlap_left_dis, overlap_right_dis = Fun3_overlap_heatpipe(
        x, heatpipe, component, safe_dis=heatpipe.width
    )

    comp_intensity = copy.deepcopy(component.intensity_backup).reshape(-1, 1)

    # 考虑左板
    num1 = np.sum(component.comp_num_plane[0:2])
    comp_left_intensity = comp_intensity[0:num1, :]

    # 根据每个组件实际跨越几根热管 平均分配热耗
    overlapflag_left = np.zeros_like(overlap_left)
    overlapflag_left[overlap_left > 0] = 1
    comp_in_hep_num = np.sum(overlapflag_left, axis=0).reshape(1, -1)  # 1x66

    comp_hep_dis_left = np.zeros((1, num1))
    if (comp_in_hep_num.size > 0) and (np.min(comp_in_hep_num) < 1):  # 说明存在组件没有骑在热管上
        # 1x66 返回没有骑在热管上组件 距其最近热管的距离 >0
        temp = -np.max(overlap_left_dis, axis=0).reshape(1, -1)
        # 1x66 logicals 没有骑在热管上组件的编号为1时 代表没有骑在热管上
        comp_index_left = comp_in_hep_num < 1
        comp_hep_dis_left[comp_index_left] = temp[comp_index_left]  # 1x66

    # 平均分配功耗
    comp_left_intensity_real = copy.deepcopy(comp_left_intensity.reshape(1, -1))
    comp_left_intensity_real[comp_in_hep_num > 0] = (
        comp_left_intensity_real[comp_in_hep_num > 0]
        / comp_in_hep_num[comp_in_hep_num > 0]
    )

    # 计算每根热管上的总功耗
    hep_power_left_matrix = (
        comp_left_intensity_real.repeat(heatpipe.number, axis=0)
    ) * overlapflag_left  # 16x66
    hep_power_left = np.sum(hep_power_left_matrix, axis=1)  # 16x1

    # 考虑右板
    num2 = np.sum(component.comp_num_plane)
    comp_right_intensity = comp_intensity[num1::, :]

    overlapflag_right = np.zeros_like(overlap_right)
    overlapflag_right[overlap_right > 0] = 1
    comp_in_hep_num = np.sum(overlapflag_right, axis=0).reshape(1, -1)  # 1x53

    comp_hep_dis_right = np.zeros((1, num2 - num1))
    if (comp_in_hep_num.size > 0) and (np.min(comp_in_hep_num) < 1):  # 说明存在组件没有骑在热管上
        # 1x53 返回没有骑在热管上组件 距其最近热管的距离 >0
        temp = -np.max(overlap_right_dis, axis=0).reshape(1, -1)
        # 1x53 logicals 没有骑在热管上组件的编号为1时 代表没有骑在热管上
        comp_index_right = comp_in_hep_num < 1
        comp_hep_dis_right[comp_index_right] = temp[comp_index_right]  # 1x53

    # 平均分配功耗
    comp_right_intensity_real = copy.deepcopy(comp_right_intensity.reshape(1, -1))
    comp_right_intensity_real[comp_in_hep_num > 0] = (
        comp_right_intensity_real[comp_in_hep_num > 0]
        / comp_in_hep_num[comp_in_hep_num > 0]
    )

    # 计算每根热管上的总功耗
    hep_power_right_matrix = (
        comp_right_intensity_real.repeat(heatpipe.number, axis=0)
    ) * overlapflag_right  # 16x66
    hep_power_right = np.sum(hep_power_right_matrix, axis=1)  # 16x1

    # 得到返回的变量
    # 左板上每根热管承载的总功耗，右板上每根热管承载的总功耗
    hep_power = np.vstack((hep_power_left.T, hep_power_right.T))  # 返回 2x16
    #  > 0 表示不干涉 距离最近热管的距离，=0 表示存在干涉
    comp_hep_dis = np.vstack((comp_hep_dis_left.T, comp_hep_dis_right.T))  # 返回 119x1
    return (hep_power, comp_hep_dis)


def Interpreter(x_opt, component):
    """将优化变量x_opt转化为所有组件对应的位置向量x_com

    Args:
        x_opt: optimization variable
        component: class-Component

    Returns:
        x_com: 1x238 [x1, y1, ..., x119, y119]
    """
    # 确保 x_opt为 1x166
    if x_opt.ndim == 1:
        x_opt = x_opt.reshape(1, -1)

    comp_position = x_opt.reshape(-1, 2)  # 119x2

    x_com = comp_position.reshape(1, -1)

    return x_com


def getObjectiveConstraint(x, domain, component, heatpipe):
    """
    根据输入组件的位置向量，计算组件和热管的干涉情况

    Args:
        x: position variable (dim: 1x2n)
        domain: Parameters中设置参数
        heatpipe: Parameters中设置参数
        component: Parameters中设置参数

    Returns:
        objective: list = [obj1]
        constraint: list = [cons1, cons2, cons3, cons4] (require: <=0)
    """

    # 计算热管相关特性
    hep_power, comp_hep_dis = Fun3_heatpower(x, heatpipe, component)

    # obj 1：heatpower
    obj = np.max(hep_power)

    # cons 1：不干涉约束
    overlap = Fun1_overlap(x, domain, component)
    cons1 = overlap

    # 约束 2：质心优化目标
    yc = Fun2_systemcentroid(x, component)
    yc_ex = 0
    dyc_ex = 0.5
    cons2 = np.max([np.abs(yc - yc_ex) - dyc_ex, 0])

    # 约束 3：单根热管总功耗约束 不超过120 W (实际可到135W)
    hep_maxload = heatpipe.maxload
    cons3 = np.sum(hep_power[hep_power > hep_maxload] - hep_maxload)

    # 约束 4：组件必须和任意一根热管相交
    cons4 = np.sum(comp_hep_dis)

    objective = [obj]
    constraint = [cons1, cons2, cons3, cons4]
    return (objective, constraint)


def Funfitness(x_opt, domain, component, heatpipe):
    """根据优化目标和优化约束构建合适的适应度函数"""

    x_com = Interpreter(x_opt, component)

    objective, constraint = getObjectiveConstraint(x_com, domain, component, heatpipe)

    obj = objective
    cons1, cons2, cons3, cons4 = constraint

    fitness = obj + cons1 + cons2 + cons3 + cons4
    return fitness


def DisplayResults(x, domain, component, heatpipe):
    """
    根据输入组件的位置向量，计算组件和热管的干涉情况

    Args:
        x: position variable (dim: 1x2n)
        domain: Parameters中设置参数
        heatpipe: Parameters中设置参数
        component: Parameters中设置参数

    Returns:
        None
    """
    objective, constraint = getObjectiveConstraint(x, domain, component, heatpipe)
    _ = objective
    cons1, cons2, cons3, cons4 = constraint

    # 计算热管相关特性
    hep_power, comp_hep_dis = Fun3_heatpower(x, heatpipe, component)

    print("The overlap volume: ", cons1)
    print("The maximum heatpipe load (left):  ", hep_power[0, :])
    print("The maximum heatpipe load (right): ", hep_power[1, :])
    print()

    print("Objective:  \n", objective)
    print("Constraint: \n", constraint)


def plot_layout(
    x,
    domain,
    component,
    heatpipe,
    prefix_name="temp",
    savefig=False,
    disfig=True,
    dismass=False,
):
    """画出布局图

    Args:
        x: position variable (dim: 1x2n)
    """
    comp_position = x.reshape(-1, 2)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    # 画出布局区域边界
    width = domain.subdomain1.length
    height = domain.subdomain1.width
    dom_left_dw_pos = (domain.location[0, 0], domain.location[0, 1])
    # dom_left_up_pos = (domain.location[1, 0], domain.location[1, 1])
    rect_dom1 = plt.Rectangle(dom_left_dw_pos, width, height, fill=False, color="red")
    # rect_dom2 = plt.Rectangle(dom_left_up_pos, width, height, fill=False, color="red")
    ax.add_patch(rect_dom1)
    # ax.add_patch(rect_dom2)

    # 画出左板上的所有组件
    num1 = sum(component.comp_num_plane[0:2])
    comp_left_loc = comp_position[0:num1, :] - component.comp_size[0:num1, :] / 2
    comp_left_width = component.comp_size[0:num1, 0]
    comp_left_height = component.comp_size[0:num1, 1]
    for i in range(num1):
        rect_comp = plt.Rectangle(
            comp_left_loc[i, :],
            comp_left_width[i],
            comp_left_height[i],
            fill=False,
            color="blue",
            linewidth=0.5,
        )
        ax.add_patch(rect_comp)
        if dismass:
            plt.text(
                comp_left_loc[i, 0],
                comp_left_loc[i, 1] + comp_left_height[i] / 2,
                f"{i+1}, {component.mass[i, 0]}kg",
                fontsize=6,
            )
        else:
            plt.text(
                comp_left_loc[i, 0],
                comp_left_loc[i, 1] + comp_left_height[i] / 2,
                f"{i+1}",
                fontsize=6,
            )

    # 画出左板上的所有热管
    hep_num = heatpipe.number
    hep_loc = heatpipe.position - np.tile(heatpipe.rec_size, (hep_num, 1)) / 2
    for i in range(hep_num):
        rect_hep = plt.Rectangle(
            hep_loc[i, :],
            heatpipe.rec_size[0],
            heatpipe.rec_size[1],
            fill=False,
            color="lime",
            linewidth=0.5,
            linestyle="dashed",
        )
        ax.add_patch(rect_hep)

    # plt.axis('equal')
    x_lb = domain.location[0, 0] * 1.05
    x_ub = domain.location[0, 2] * 1.05
    y_lb = domain.location[0, 1] * 1.05
    y_ub = domain.location[0, 3] * 1.05
    plt.axis([x_lb, x_ub, y_lb, y_ub])

    if savefig:
        left_name = prefix_name + "_left.jpg"  # "Layout_left_opt3.jpg"
        plt.savefig(left_name, dpi=300, bbox_inches="tight")
    if disfig:
        plt.show()


if __name__ == "__main__":
    import time

    from Param import domain, component, heatpipe

    t1 = time.time()
    # x = component.original_position.reshape(1, -1)
    dim = component.variable_num
    for _ in range(300):
        x_opt = (
            np.random.rand(1, dim) * (component.x_max - component.x_min)
            + component.x_min
        )
        # x_opt = x[component.variable_indicator > 0].reshape(1, -1)
        fitness = Funfitness(x_opt, domain, component, heatpipe)
        print("fitness value: ", fitness)  # 223.3648
    print("Running time for single evaluation: ", time.time() - t1)

    # display results
    DisplayResults(x_opt, domain, component, heatpipe)

    # draw layout
    plot_layout(x_opt, domain, component, heatpipe, savefig=False, disfig=True, dismass=False)
