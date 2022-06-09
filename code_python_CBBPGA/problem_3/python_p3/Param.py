# -*- encoding: utf-8 -*-
'''
@File    :   Param.py
@Time    :   2021/11/26 15:18:44
@Author  :   Xianqi
@Contact :   chenxianqi12@qq.com
'''


import numpy as np
import json


class RecDomainRepresentation:
    def __init__(self, location):
        """
        输入左下角顶点坐标和右上角顶点坐标
        location: [x1, y1, x2, y2]
        输出四条边对应矩形的顶点坐标和尺寸
        """
        self.extend_length = 1000
        self.location = location
        self.rec_number = 4
        self.width = self.location[2] - self.location[0]
        self.height = self.location[3] - self.location[1]
        self.rec_position = np.array(
            [
                [
                    self.location[0] / 2 + self.location[2] / 2,
                    self.location[1] - self.extend_length / 2,
                ],
                [
                    self.location[0] - self.extend_length / 2,
                    self.location[1] / 2 + self.location[3] / 2,
                ],
                [
                    self.location[0] / 2 + self.location[2] / 2,
                    self.location[3] + self.extend_length / 2,
                ],
                [
                    self.location[2] + self.extend_length / 2,
                    self.location[1] / 2 + self.location[3] / 2,
                ],
            ]
        )
        self.rec_size = np.array(
            [
                [self.width + 2 * self.extend_length, self.extend_length],
                [self.extend_length, self.height],
                [self.width + 2 * self.extend_length, self.extend_length],
                [self.extend_length, self.height],
            ]
        )
        self.rec_angle = np.zeros((self.rec_number, 1))


class TelSatBoardDomain:
    """通信卫星所有布局板描述类"""
    def __init__(self, name="TelSat"):
        self.name = name

        self.part = []  # 定义每个子部分：left, right, middle
        self.part_num = 0

        self.comp_id = []
        self.comp_num = len(self.comp_id)

        self.hp_id = []
        self.hp_num = len(self.hp_id)

    def load_TelSatLayoutDomain(self, telsatlayoutdomain):
        assert isinstance(telsatlayoutdomain, TelSatLayoutDomain)

        self.part.append(telsatlayoutdomain)
        self.part_num = len(self.part)

        self.comp_id = self.comp_id + telsatlayoutdomain.comp_id
        self.comp_num = len(self.comp_id)

        self.hp_id += telsatlayoutdomain.hp_id
        self.hp_num = len(self.hp_id)

        if self.part_num > 1:
            if self.part[0].name == 'right':
                # 说明不是按照 left - right 的顺序进行排列的
                self._switch_telsatlayoutdomain_in_part()

    def _switch_telsatlayoutdomain_in_part(self):
        # 调换 part 中的数据顺序
        if self.part_num > 1:  # 当存在左右两个布局板时才可以进行更换
            temp = self.part[0]
            self.part[0] = self.part[1]
            self.part[1] = temp

            # 更新组件的顺序
            self.comp_id = []
            self.comp_id = self.comp_id + self.part[0].comp_id + self.part[1].comp_id
            self.comp_num = len(self.comp_id)

            # 更新热管的顺序
            self.hp_id = []
            self.hp_id = self.hp_id + self.part[0].hp_id + self.part[1].hp_id
            self.hp_num = len(self.hp_id)


class TelSatLayoutDomain:
    """通信卫星可布局南北板描述类"""
    def __init__(self, location, comps=[], heatpipes=[], name=None):
        """
        Args:
            location: [numpy.ndarray] (dim: 1x4) [[x1, y1, x2, y2]] 矩形左下角顶点坐标(x1, y1)和右上角顶点坐标(x2, y2)
            comps: [list] 包含所有组件ID的列表
            heatpipes: [list] 包含所有热管ID的列表
            name: [str] 该布局区域的名字  "left" / "right"
        """
        assert isinstance(location, np.ndarray), "Not supported data type"
        assert isinstance(comps, list), "Not supported data type"
        assert isinstance(heatpipes, list), "Not supported data type"

        self.location = location.reshape(1, -1)
        # width: 布局板的宽 (沿着x方向的长度 - 卫星坐标系中X方向)
        self.width = self.location[0, 2] - self.location[0, 0]
        # height: 布局板的高 (沿着y方向的长度 - 卫星坐标系中Z方向)
        self.height = self.location[0, 3] - self.location[0, 1]

        self.comp_id = comps
        self.comp_num = len(self.comp_id)

        self.hp_id = heatpipes
        self.hp_num = len(self.hp_id)

        self.name = name

        # 定义南北板的布局区域部分：part
        self.part = []  # 定义每个子区域 down, up
        self.part_num = 0

    def load_LayoutDomain(self, layoutdomain):
        """载入南北板上的上下两部分布局区域"""
        assert isinstance(layoutdomain, LayoutDomain)

        self.part.append(layoutdomain)
        self.part_num = len(self.part)

        self.comp_id = self.comp_id + layoutdomain.comp_id
        self.comp_num = len(self.comp_id)

        if self.hp_num == 0:
            self.hp_id = layoutdomain.hp_id
            self.hp_num = layoutdomain.hp_num

        # 确保上下两个板的数据按照 [down, up] 的顺序存放
        if self.part_num == 2:
            if self.part[0].name == 'up':
                # 表示需要更换顺序
                self._switch_layoutdomain_in_part()

    def _switch_layoutdomain_in_part(self):
        # 调换 part 中数据的顺序
        if self.part_num == 2:  # 当存在上下两个布局区域时才可以进行更换
            temp = self.part[0]
            self.part[0] = self.part[1]
            self.part[1] = temp

            # 更新组件的顺序
            self.comp_id = []
            self.comp_id = self.comp_id + self.part[0].comp_id + self.part[1].comp_id
            self.comp_num = len(self.comp_id)


class LayoutDomain:
    """矩形布局区域描述类"""
    def __init__(self, location, comps=[], heatpipes=[], name=None):
        """
        Args:
            location: [numpy.ndarray] (dim: 1x4) [[x1, y1, x2, y2]] 矩形左下角顶点坐标(x1, y1)和右上角顶点坐标(x2, y2)
            comps: [list] 包含所有组件ID的列表
            heatpipes: [list] 包含所有热管ID的列表
            name: [str] 该布局区域的名字 "up" / "down"
        """
        assert isinstance(location, np.ndarray), "Not supported data type"
        assert isinstance(comps, list), "Not supported data type"
        assert isinstance(heatpipes, list), "Not supported data type"

        self.location = location.reshape(1, -1)
        self.width = self.location[0, 2] - self.location[0, 0]
        self.height = self.location[0, 3] - self.location[0, 1]

        self.comp_id = comps
        self.comp_num = len(self.comp_id)

        self.hp_id = heatpipes
        self.hp_num = len(self.hp_id)

        self.name = name

        # 载入该布局区域的矩形表征
        self.recDomRep = RecDomainRepresentation(self.location.reshape(-1))


class SingleComponent:
    def __init__(
        self,
        name,
        width,
        height,
        mass,
        intensity,
        heatpipe_num,
        origin,
        backups,
        dom,
        heatpipe_range,
        location_range
    ):
        """
        Args:
            name: 每个组件的 id
            type: 组件的类型
            width: 宽度  mm
            height: 高度  mm
            mass: 质量 kg
            intensity: 热功率 W
            heatpipe_num: 组件占据热管的数量
            origin: 备份组件的源组件
            backups: 组件的备份组件
            dom: 组件所摆放的布局区域
            heatpipe_range: 组件所允许占据热管的编号范围
            location_range: 组件所允许摆放位置的位置
        """
        self.name = name
        self.width = width
        self.height = height
        self.mass = mass
        self.intensity = intensity
        self.hp_num = heatpipe_num
        self.origin = origin
        self.backups = backups
        self.dom = dom
        self.heatpipe_range = np.array(heatpipe_range)  # [min, max]
        self.location_range = np.array(location_range)  # [xmin, xmax, ymin, ymax]


class Component:
    def __init__(self, comp_num):
        """涵盖所有组件信息的类
        Args:
            comp_num: 组件的数量
        """
        self.number = comp_num
        self.comp_num = comp_num

        self.name = self.comp_num * [0]
        self.name_index = {}  # 存储字典，name为keys，索引index为value
        self.comp_size = np.zeros((self.comp_num, 2))
        self.comp_angle = np.zeros((self.comp_num, 1))  # 默认组件角度为 0
        self.comp_mass = np.zeros((self.comp_num, 1))
        self.comp_intensity = np.zeros((self.comp_num, 1))
        self.comp_occ_hp_num = np.zeros((self.comp_num, 1))  # 组件占据热管数量
        self.comp_num_in_plane = np.zeros((4,))  # 一共 4 块布局板

        # 定义组件占据热管编号
        self.comphp_index = np.zeros((self.comp_num, 1))
        self.comphp_index_min = np.zeros((self.comp_num, 1))  # 组件占据热管编号的最小值
        self.comphp_index_max = np.ones((self.comp_num, 1))  # 组件占据热管编号的最大值 (需要后续重新设置)

        # 定义组件位置
        self.location = np.zeros((self.comp_num, 2))
        self.location_min_default = np.zeros((self.comp_num, 2))  # [x1min, y1min; ...; xnmin, ynmin]
        self.location_max_default = np.zeros((self.comp_num, 2))  # [x1max, y1max; ...; xnmax, ynmax]
        self.location_min = self.location_min_default.copy()
        self.location_max = self.location_max_default.copy()

        self.board = self.comp_num * [0]

        # 载入 备份组件 与 原始组件
        self.origin = self.comp_num * [0]
        self.backups = self.comp_num * [0]

        # 设置和备份组件相关的变量
        self.backup_indicator = np.ones((self.comp_num, 1))
        self.comp_intensity_backup = self.comp_intensity.copy()
        self.backup_distance = 20  # 设置备份组件摆放间隔

        # 设置优化变量
        self.loc_opt_indicator = np.ones((self.comp_num, 2))
        self.x_opt_dim = 2 * self.comp_num
        self.x_opt_min = None
        self.x_opt_max = None

        # 设置分组变量
        self.group = comp_num * [0]
        self.groups = {}  # 存储数据中的groups信息

        # 设置组件是否锁定的状态变量
        self.lock_indicator = np.zeros((self.comp_num, 2))

    def load_singlecomponent(
        self,
        index,
        name,
        comp_width,
        comp_height,
        comp_mass,
        comp_intensity,
        comp_occ_hp_num,
        origin,
        backups,
        backup_distance,
        place_dom,  # 摆放的布局区域名称
        dom_location,  # 可摆放区域的坐标 [xmin, ymin, xmax, ymax]
        dom_heatpipe_num,  # 可摆放区域的热管数量
        location_range,  # [xmin, xmax, ymin, ymax]
        heatpipe_range,  # 可以是列表 [hp_min, hp_max]
        comp_group,  # str
        comp_location, # 导入json文件中的位置
        islocked,  # True or False
    ):
        assert isinstance(name, str), "Not supported data type for <name>."
        for i in range(len(location_range)):
            if location_range[i] is None:
                location_range[i] = - float('inf') if (i == 0) or (i == 2) else float('inf')
        
        for i in range(len(heatpipe_range)):
            if heatpipe_range[i] is None:
                heatpipe_range[i] = - float('inf') if i == 0 else float('inf')

        self.name[index] = name
        self.name_index[name] = index
        self.comp_size[index, :] = np.array([comp_width, comp_height])
        self.comp_mass[index, 0] = comp_mass
        self.comp_intensity[index, 0] = comp_intensity
        self.comp_occ_hp_num[index, 0] = comp_occ_hp_num

        self.origin[index] = origin  # 设置备份组件相关参数
        self.backups[index] = backups

        self.board[index] = place_dom  # 应该是字符串

        # 设置组件摆放位置坐标的变化范围
        self.location_min_default[index, :] = dom_location.reshape(-1)[0:2] + self.comp_size[index, :] / 2
        self.location_max_default[index, :] = dom_location.reshape(-1)[2:4] - self.comp_size[index, :] / 2
        location_min_set = np.array([location_range[0], location_range[2]])
        location_max_set = np.array([location_range[1], location_range[3]])
        self.location_min[index, :] = np.maximum(self.location_min_default[index, :], location_min_set)
        self.location_max[index, :] = np.minimum(self.location_max_default[index, :], location_max_set)

        # 设置可摆放热管编号的变化范围
        self.comphp_index_max[index, 0] = int(dom_heatpipe_num - 1)
        comhp_index_min_set = heatpipe_range[0]
        comhp_index_max_set = heatpipe_range[1]
        self.comphp_index_min[index, 0] = int(np.maximum(self.comphp_index_min[index, 0], comhp_index_min_set))
        self.comphp_index_max[index, 0] = int(np.minimum(self.comphp_index_max[index, 0], comhp_index_max_set))

        # 设置备份组件指示变量
        self.backup_distance = backup_distance
        # TODO 需要在未来进行改进，更好地表示备份组件及其相关组件分组信息
        # TODO 后台手动更改选定 备份组件组 时 应该自动填入 origin 和 backups
        if origin == "":
            pass
        else:
            # 表示为备份组件
            self.backup_indicator[index, 0] = 0
            self.loc_opt_indicator[index, :] = np.zeros((1, 2))
        
        # 载入 comp_group 变量
        self.group[index] = comp_group

        # 载入 comp_location 变量
        self.location[index, :] = np.array(comp_location)

        # 载入 lock_indicator 变量
        if islocked:
            # 锁定状态下 位置不可进行优化 热管编号同时也应该固定
            self.lock_indicator[index, :] = np.ones((1, 2))
            self.loc_opt_indicator[index, :] = np.zeros((1, 2))

    def interpreter_x_opt2pos(self, x_opt):
        """将优化变量转换为组件的位置变量
        
        Args:
            x_opt: 1x2m

        Returns:
            x_com: 1x2n
        """
        opt_indicator = self.loc_opt_indicator.copy().reshape(1, -1)
        x_com = np.zeros((1, 2 * self.comp_num))
        x_com[opt_indicator > 0] = x_opt.reshape(-1).copy()
        comp_location = x_com.reshape(-1, 2)

        # 备份组件处理 (假定备份组件从上到下排列，优化最上面的组件位置)
        # 1. 根据备份组件信息 生成转换矩阵 M[x_com.dim, x_opt.dim]
        # 2. y 坐标的偏移 delta_location
        transM = np.zeros((int(x_com.size / 2), int(x_opt.size / 2)))
        delta_location = np.zeros_like(comp_location)
        backup_ind = [0] * self.comp_num

        transM[self.loc_opt_indicator[:, 0] > 0, :] = np.eye(int(x_opt.size / 2))
        for i in range(self.comp_num):
            if self.origin[i] == "":
                pass
            else:
                name = self.origin[i]
                index = self.name_index[name]
                backup_ind[index] += 1
                transM[i, :] = transM[index, :]
                delta_location[i, 1] = - backup_ind[index] * (self.comp_size[i, 1] + self.backup_distance)
        # 3. 求出备份组件坐标
        comp_location = np.dot(transM, x_opt.reshape(-1, 2))
        comp_location += delta_location

        # 固定位置组件坐标处理
        comp_location[self.lock_indicator > 0] = self.location[self.lock_indicator > 0]
        x_com = comp_location.reshape(1, -1)
        return x_com
    
    def interpreter_x_pos2opt(self, x_com):
        """将组件的位置变量转换为优化变量
        
        Args:
            x_com: 1x2n

        Returns:
            x_opt: 1x2m
        """
        x_opt = x_com[self.loc_opt_indicator.reshape(1, -1) > 0]
        return x_opt

    def update(self, groups={}, comp_num_in_plane=None):
        if comp_num_in_plane is not None:
            # 更新每个布局区域上的组件数量
            self.comp_num_in_plane = np.array([comp_num_in_plane[i] for i in range(len(comp_num_in_plane))])

        # 根据groups信息更新self.backup_indicator
        self.groups = groups
        for key in groups.keys():
            count = 1
            comps_work_num = groups[key]['comps_work_num']
            for name in groups[key]['comps']:
                ind = self.name_index[name]
                if self.backup_indicator[ind] == 0:
                    if count < comps_work_num:
                        # 选择第一个为0的备份，根据其 comps_work_num 调为正常工作
                        self.backup_indicator[ind] = 1
                        count += 1
        
        # 更新备份组件功率
        self.comp_intensity_backup = self.comp_intensity * self.backup_indicator

        # 更新优化变量设置
        self.x_opt_dim = np.sum(self.loc_opt_indicator).astype(int)
        self.x_opt_min = self.location_min[self.loc_opt_indicator > 0].reshape(1, -1)
        self.x_opt_max = self.location_max[self.loc_opt_indicator > 0].reshape(1, -1)

        # 整数化
        self.comphp_index_min = self.comphp_index_min.astype(int)
        self.comphp_index_max = self.comphp_index_max.astype(int)
        self.comp_occ_hp_num = self.comp_occ_hp_num.astype(int)

        # Only for competition
        # v2.0 : reduce the height by its half
        self.comp_size[:, 1] = self.comp_size[:, 1] / 2
        
        # TODO 需要根据备份组件信息来进一步缩小备份组件的优化变量范围
        # TODO 需要根据给定热管范围来更新所有组件的可放置位置范围


class HeatPipe():
    def __init__(self, heatpipe_num):
        self.number = heatpipe_num
        self.hp_num = heatpipe_num
        self.hp_num_in_plane = np.zeros((2, ))

        self.name = self.hp_num * [0]
        self.name_index = {}
        self.hp_size = np.zeros((self.hp_num, 2))
        self.hp_maxload = np.zeros((self.hp_num,))
        self.location = np.zeros((self.hp_num, 2))
        self.board = self.hp_num * [0]
        self.width = 0
        self.height = 0

    def load_singleheatpipe(self, index, name, width, height, maxload, location, board):
        self.name[index] = name
        self.name_index[name] = index
        self.hp_size[index, :] = np.array([width, height])
        self.hp_maxload[index] = maxload
        self.location[index, :] = np.array([location[0], location[1]])
        self.board[index] = board

    def update(self, hp_num_in_plane=None):
        if hp_num_in_plane is not None:
            # 更新每个布局区域上的热管数量
            self.hp_num_in_plane = np.array([hp_num_in_plane[i] for i in range(len(hp_num_in_plane))])
        
        self.width = self.hp_size[0, 0]
        self.height = self.hp_size[0, 1]


class Route():  # TODO
    def __init__(self, route_num):
        self.number = route_num
        self.route_num = route_num

        self.routes = {}
        self.routes_node_num = self.route_num * [0]
        self.routes_max_length = np.zeros((self.route_num,))
        self.route_connect = None

        self.horizontal_factor = 0.5
        self.vertical_factor = 1 - self.horizontal_factor
        self.routes_max_length_horizontal = None
        self.routes_max_length_vertical = None

    def load_route(self, routes, route_connect, routes_max_length):
        """
        Args:
            routes: [list] 每条路径表示一个元素
            route_connnect: [dict] 字典 keys为组件的ID (name), 值为下一个连接的组件ID
            routes_max_length: [list] 每条路径允许的最大链路长度
        """
        num = len(routes)
        assert num == self.number
        for i in range(num):
            self.routes[i] = routes[i]
            self.routes_node_num[i] = len(routes[i])
            self.routes_max_length[i] = routes_max_length[i]
        self.route_connect = route_connect

    def set_route_length_horizontal_factor(self, horizontal_factor=None):
        # 设置水平链路所允许的最大链路长度
        if horizontal_factor is not None:
            self.horizontal_factor = horizontal_factor
        self.routes_max_length_horizontal = self.horizontal_factor * self.routes_max_length

    def set_route_length_vertical_factor(self, vertical_factor=None):
        if vertical_factor is not None:
            self.vertical_factor = vertical_factor
        self.routes_max_length_vertical = self.vertical_factor * self.routes_max_length


class Group():  # TODO
    def __init__(self, group_num):
        self.number = group_num
        self.group_num = group_num
        self.name = group_num * [0]
        self.groups = {}
    
    def load_groups(self, groups):
        for key in groups.keys():
            self.groups[key] = groups[key]


class JsonDecode():
    """将json文件解析为不同的数据类，供后续使用"""
    def __init__(self, json_data):
        self.data = json_data
        self.domain = None
        self.component = None
        self.heatpipe = None
        self.route = None

    def decode_domain(self):
        telSatBoardDomain = TelSatBoardDomain(name="TelSat")

        # 导入左板上
        layout_dom = {}
        name = ['down', 'up', 'down', 'up']
        layout_dom[0] = self.data['dom']['left']['down']
        layout_dom[1] = self.data['dom']['left']['up']
        layout_dom[2] = self.data['dom']['right']['down']
        layout_dom[3] = self.data['dom']['right']['up']

        layout_domain = {}
        for i in range(4):
            location = layout_dom[i]['location']
            location = np.array(location)
            heatpipes = layout_dom[i]['heatpipe']
            comps = layout_dom[i]['comp']

            layout_domain[i] = LayoutDomain(location, comps, heatpipes, name=name[i])

        # 载入左板
        location_left = np.array(self.data['dom']['left']['location'])
        telSatLayoutDomain1 = TelSatLayoutDomain(location_left, name='left')
        telSatLayoutDomain1.load_LayoutDomain(layout_domain[0])
        telSatLayoutDomain1.load_LayoutDomain(layout_domain[1])

        # 载入右板
        location_right = np.array(self.data['dom']['right']['location'])
        telSatLayoutDomain2 = TelSatLayoutDomain(location_right, name='right')
        telSatLayoutDomain2.load_LayoutDomain(layout_domain[2])
        telSatLayoutDomain2.load_LayoutDomain(layout_domain[3])

        telSatBoardDomain.load_TelSatLayoutDomain(telSatLayoutDomain1)
        telSatBoardDomain.load_TelSatLayoutDomain(telSatLayoutDomain2)

        self.domain = telSatBoardDomain
        return self.domain

    def decode_component(self):
        comp_num = self.domain.comp_num
        component = Component(comp_num)

        comp_index = 0
        comp_num_in_plane = []
        for board_ind in range(2):
            # 遍历 左板 和 右板
            board_domain = self.domain.part[board_ind]
            for laydom_ind in range(2):
                # 遍历 下板 和 上板
                layout_domain = board_domain.part[laydom_ind]
                comp_num_in_plane.append(layout_domain.comp_num)
                for comp_name in layout_domain.comp_id:
                    comp = self.data['comps'][comp_name]
                    width = comp['size']['width']
                    height = comp['size']['height']
                    mass = comp['mass']
                    intensity = comp['intensity']
                    comp_occ_hp_num = comp['heatpipe_num']
                    origin = comp['origin']
                    backups = comp['backups']
                    backup_distance = self.data['global']['backup_distance']
                    place_dom = comp['board']
                    dom_location = layout_domain.location
                    dom_heatpipe_num = layout_domain.hp_num
                    location_range = comp['location_range']
                    heatpipe_range = comp['heatpipe_range']
                    comp_group = comp['group']
                    comp_location = comp['location']
                    islocked = comp['islocked']
                    component.load_singlecomponent(
                        comp_index,
                        comp_name,
                        width,
                        height,
                        mass,
                        intensity,
                        comp_occ_hp_num,
                        origin,
                        backups,
                        backup_distance,
                        place_dom,
                        dom_location,
                        dom_heatpipe_num,
                        location_range,
                        heatpipe_range,
                        comp_group,
                        comp_location,
                        islocked,
                    )

                    comp_index += 1

        groups = self.data['groups']
        component.update(groups=groups, comp_num_in_plane=comp_num_in_plane)

        self.component = component
        return self.component

    def decode_heatpipe(self):
        hp_num = self.domain.hp_num
        heatpipe = HeatPipe(hp_num)

        hp_index = 0
        name_pres = ['L', 'R']  # TODO --> 修改为keys中自动分割字符串
        hp_num_in_plane = []
        for board_ind in range(2):
            board_domain = self.domain.part[board_ind]
            name_pre = name_pres[board_ind]
            board_hp_num = board_domain.hp_num
            hp_num_in_plane.append(board_hp_num)
            for i in range(board_hp_num):
                name = name_pre + str(i)
                index = hp_index
                width = self.data['heatpipes'][name]['width']
                height = self.data['heatpipes'][name]['height']
                maxload = self.data['heatpipes'][name]['maxload']
                location = [
                    self.data['heatpipes'][name]['location']['x'],
                    self.data['heatpipes'][name]['location']['y'],
                ]
                board = self.data['heatpipes'][name]['board']
                heatpipe.load_singleheatpipe(
                    index,
                    name,
                    width,
                    height,
                    maxload,
                    location,
                    board,
                )

                hp_index += 1

        heatpipe.update(hp_num_in_plane)
        self.heatpipe = heatpipe
        return self.heatpipe

    def decode_route(self):
        route_num = self.data['route']['route_num']
        route = Route(route_num)

        routes = self.data['route']['routes']
        route_connect = self.data['route']['route_connect']
        routes_max_length = self.data['route']['routes_max_length']
        route.load_route(routes, route_connect, routes_max_length)

        horizontal_factor = self.data['route']['horizontal_factor']
        vertical_factor = self.data['route']['vertical_factor']
        route.set_route_length_horizontal_factor(horizontal_factor)
        route.set_route_length_vertical_factor(vertical_factor)

        self.route = route
        return self.route


if __name__ == "__main__":
    json_file = 'test_v0.json'
    f = open(json_file, encoding="utf-8")
    json_data = json.load(f)
    jd = JsonDecode(json_data)

    domain = jd.decode_domain()
    print(domain.comp_num)
    component = jd.decode_component()
    heatpipe = jd.decode_heatpipe()
    route = jd.decode_route()
