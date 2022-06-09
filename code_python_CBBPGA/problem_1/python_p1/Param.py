import numpy as np
import scipy.io as sci


class SubDomain:
    def __init__(self, location):
        """
        输入左下角顶点坐标和右上角顶点坐标
        location: [x1, y1, x2, y2]
        输出四条边对应矩形的顶点坐标和尺寸
        """
        self.extend_length = 100
        self.location = location
        self.rec_number = 4
        self.length = self.location[2] - self.location[0]
        self.width = self.location[3] - self.location[1]
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
                [self.length + 2 * self.extend_length, self.extend_length],
                [self.extend_length, self.width],
                [self.length + 2 * self.extend_length, self.extend_length],
                [self.extend_length, self.width],
            ]
        )
        self.rec_angle = np.zeros((self.rec_number, 1))


class Domain:
    def __init__(self):
        self.length = 80
        self.width = 50
        self.location = np.array(
            [
                [-40, -25, 40, 25],  # 左板下 两个顶点 (x1, y1, x2, y2) 坐标
                [0, 0, 0, 0],  # 左板上
                [0, 0, 0, 0],  # 右板下
                [0, 0, 0, 0],  # 右板上
            ]
        )
        self.subdomain1 = SubDomain(self.location[0, :])
        self.subdomain2 = SubDomain(self.location[1, :])
        self.subdomain3 = SubDomain(self.location[2, :])
        self.subdomain4 = SubDomain(self.location[3, :])


class Component:
    def __init__(self, comp_infor, domain):
        self.number = comp_infor.shape[0]
        self.comp_size = comp_infor[:, 1:3].reshape(-1, 2)
        self.mass = comp_infor[:, 3].reshape(-1, 1)
        self.intensity = comp_infor[:, 4].reshape(-1, 1)
        self.comp_num_plane = np.array([self.number, 0, 0, 0])  # 每个板上的组件数
        self.angle = np.zeros((self.number, 1))
        self.backup_indicator = np.ones((self.number, 1))  # 0 表示为备份组件
        self.backup_distance = 20  # 设置备份组件的距离
        self.intensity_backup = self.intensity.copy()
        self.set_optimization_variable()
        self.set_variable_range(domain)

    def set_optimization_variable(self):
        variable_indicator = np.ones((self.number, 2))
        self.variable_indicator = variable_indicator.reshape(1, -1).astype(
            "int"
        )  # 1 x 2n
        self.x_opt_dim = np.sum(self.variable_indicator)

    def set_variable_range(self, domain):
        self.pos_x_min = domain.location[0, 0] * np.ones((self.number, 1))
        self.pos_x_max = domain.location[0, 2] * np.ones((self.number, 1))
        self.pos_y_min = np.vstack(
            (
                domain.location[0, 1] * np.ones((self.comp_num_plane[0], 1)),
                domain.location[1, 1] * np.ones((self.comp_num_plane[1], 1)),
                domain.location[2, 1] * np.ones((self.comp_num_plane[2], 1)),
                domain.location[3, 1] * np.ones((self.comp_num_plane[3], 1)),
            )
        )
        self.pos_y_max = np.vstack(
            (
                domain.location[0, 3] * np.ones((self.comp_num_plane[0], 1)),
                domain.location[1, 3] * np.ones((self.comp_num_plane[1], 1)),
                domain.location[2, 3] * np.ones((self.comp_num_plane[2], 1)),
                domain.location[3, 3] * np.ones((self.comp_num_plane[3], 1)),
            )
        )
        self.pos_x_min = self.pos_x_min + self.comp_size[:, 0].reshape(-1, 1) / 2
        self.pos_x_max = self.pos_x_max - self.comp_size[:, 0].reshape(-1, 1) / 2
        self.pos_y_min = self.pos_y_min + self.comp_size[:, 1].reshape(-1, 1) / 2
        self.pos_y_max = self.pos_y_max - self.comp_size[:, 1].reshape(-1, 1) / 2

        pos_min = np.hstack((self.pos_x_min, self.pos_y_min))
        pos_max = np.hstack((self.pos_x_max, self.pos_y_max))
        self.x_min_all = pos_min.reshape(1, -1)  # 1 * 238
        self.x_max_all = pos_max.reshape(1, -1)

        # 根据实际优化变量设置上下限
        self.x_opt_min = self.x_min_all[self.variable_indicator > 0].reshape(1, -1)
        self.x_opt_max = self.x_max_all[self.variable_indicator > 0].reshape(1, -1)


class HeatPipe:
    def __init__(self):
        self.number = 4
        self.interval = 12
        self.width = 5
        self.length = 50
        self.rec_size = np.array([self.width, self.length])
        self.maxload = 30  # 每根热管的最大承载
        self.first_posx = - 40 + 14.5
        self.set_heatpipe_position()

    def set_heatpipe_position(self):
        self.position = np.zeros((self.number, 2))
        self.last_posx = self.first_posx + (self.number - 1) * (
            self.width + self.interval
        )
        posx = np.linspace(self.first_posx, self.last_posx, self.number)
        self.position[:, 0] = posx


import os


curPath = os.path.abspath(os.path.dirname(__file__))
component_information = sci.loadmat(curPath + "/excel_matrix_problem1.mat")["matrix"]
domain = Domain()
component = Component(component_information, domain)
heatpipe = HeatPipe()
