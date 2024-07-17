import plot_current_data
import train
import numpy as np
import matplotlib.pyplot as plt
import contour_error_v2
from scipy.spatial import KDTree
import sys
sys.setrecursionlimit(15000)  #
import gcode_interpreter
import copy
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

offset = [-148.55306, -295.10247]#828D

screw_lead = 10
num_joints = 2
sample_time = 0.001
pole_pair = 4

plane = 1#1为XY平面；2为YZ平面；3为XZ平面

file_path = './test_data/'
model_file_path =  './train_data/'

class FiniteStateMachine_current_movement_v2:
    '''
    '''
    def __init__(self, init_state, window_size):
        self.window_size = window_size
        self.window = []
        self.state = init_state  # 初始状态为静止
        self.rule = []

    def update_state(self, value):
        #self.window.append(value[1][0])
        self.window.append(value[1])
        if len(self.window) > self.window_size:
            self.window.pop(0)

        if self.state == 0:
            if self.window[0][0] == 2:
                self.state = 2
                self.rule.append([round(value[0], 6), value[2], '0-0_2-0'])
            elif self.window[0][0] == 3:
                self.state = 3
                if self.window[0][1] == 0:
                    self.rule.append([round(value[0], 6), value[2], '0-0_3-0'])
                elif self.window[0][1] == 1:
                    self.rule.append([round(value[0], 6), value[2], '0-0_3-1'])
            elif self.window[0][0] == 4:
                self.state = 4
                self.rule.append([round(value[0], 6), value[2], '0-0_4-0'])
        elif self.state == 1:
            if self.window[0][0] == 2:
                self.state = 2
                if self.window[0][1] == 0:
                    self.rule.append([round(value[0], 6), value[2], '1-0_2-0'])
                elif self.window[0][1] == 1:
                    self.rule.append([round(value[0], 6), value[2], '1-0_2-1'])
            elif self.window[0][0] == 3:
                self.state = 3
                self.rule.append([round(value[0], 6), value[2], '1-0_3-0'])
            elif self.window[0][0] == 4:
                self.state = 4
                self.rule.append([round(value[0], 6), value[2], '1-0_4-0'])
        elif self.state == 2:
            if self.window[0][0] == 0:
                self.state = 0
                self.rule.append([round(value[0], 6), value[2], '2-0_0-0'])
            elif self.window[0][0] == 4:
                self.state = 4
                self.rule.append([round(value[0], 6), value[2], '2-0_4-0'])
        elif self.state == 3:
            if self.window[0][0] == 1:
                self.state = 1
                self.rule.append([round(value[0], 6), value[2], '3-0_1-0'])
            elif self.window[0][0] == 4:
                self.state = 4
                self.rule.append([round(value[0], 6), value[2], '3-0_4-0'])
        elif self.state == 4:
            if self.window[0][0] == 2:
                self.state = 2
                self.rule.append([round(value[0], 6), value[2], '4-0_2-0'])
            elif self.window[0][0] == 3:
                self.state = 3
                self.rule.append([round(value[0], 6), value[2], '4-0_3-0'])

    def get_state(self):
        return self.state

    def get_rule(self):
        return self.rule

def judge_movement_current_movement_v2(sequence, pos, init_state, window_size):
    '''
    :param sequence:
    :param window_size:
    :param threshold:
    :return:
    '''
    # fsm = FiniteStateMachine_v2(start_window_size,end_window_size, threshold)
    fsm = FiniteStateMachine_current_movement_v2(init_state, window_size)

    movement = []
    for i in range(sequence.shape[1] - 1):
        #         print("时刻: ",sequence[0][i])
        fsm.update_state([sequence[0][i + 1], sequence[1][i + 1], pos[1][i + 1]])
        #         print("状态: ", fsm.get_state())
        movement.append(fsm.get_state())
    #     print(len(sequence[0]))
    #     print(len(movement))
    return np.array([sequence[0][window_size:], movement]), np.array(fsm.get_rule())

def merge_axis(rule_x, rule_y,movement):
    min_distances = []  #
    nearest_indices = []  #
    merge_t = []

    for num1 in range(len(rule_x[:, 0])):
        distances = np.abs(np.array(rule_y[:, 3], dtype=float) - float(rule_x[:, 3][num1]))

        min_distance = np.min(distances)
        nearest_index = np.argmin(distances)


        if min_distance < 100.0:
            nearest_indices.append(nearest_index)
            # rule_x[:, 0][num1] = rule_y[:, 0][nearest_index]
            merge_t.append([[float(rule_x[:, 0][num1]), float(rule_y[:, 0][nearest_index])],
                            #[rule_x[:, 1][num1], rule_y[:, 1][nearest_index]],
                            [float(rule_x[:, 3][num1]), float(rule_y[:, 3][nearest_index])],
                            #[rule_x[:, 4][num1], rule_y[:, 4][nearest_index]],
                            [rule_x[:, 2][num1], rule_y[:, 2][nearest_index]]
                            ])
            #merge_t.append([[rule_x[:, 0][num1],rule_x[:, 3][num1]],[rule_y[:, 0][nearest_index],rule_y[:, 3][nearest_index]]])
        else:
            merge_t.append([[float(rule_x[:, 0][num1]),float(rule_x[:, 0][num1])],
                            [float(rule_x[:, 3][num1]),float(rule_x[:, 3][num1])],
                            [rule_x[:, 2][num1], float(plot_current_data.get_position_at_t_v2(movement[1], float(rule_x[:, 0][num1]))[0])]
                            ])

    for i in range(len(rule_y[:,0])):
        if i not in nearest_indices:
            merge_t.append([[float(rule_y[:, 0][i]), float(rule_y[:, 0][i])],
                            [float(rule_y[:, 3][i]), float(rule_y[:, 3][i])],
                            [float(plot_current_data.get_position_at_t_v2(movement[0], float(rule_y[:, 0][i]))[0]),rule_y[:,2][i]]
                            ])

    print(len(nearest_indices))
    # merge_t = np.array(merge_t)


    def custom_key(item):
        return item[1][0]


    merge_t = sorted(merge_t, key=custom_key)

    # merge_t.sort(key=lambda x: x[1], reverse=False)

    merge_t = np.array(merge_t)
    print("merge_t.shape: ",merge_t.shape)

    return merge_t

def merge_state_4_v3(rule,movement,joint):
    '''
    :param rule:
    :return:
    '''
    rule_merged = [[rule[0][0],rule[0][1],rule[0][2],rule[0][0],rule[0][1]]]
    for i in range(rule.shape[0] - 1):
        # print('sssssss:', rule_merged[-1][2].split('_')[1])
        if rule_merged[-1][2].split('_')[1] == '4-0':
            if rule[i + 1][2].split('_')[0] == '4-0':
                if str(rule_merged[-1][2].split('_')[0]) != str(rule[i + 1][2].split('_')[1]):
                    rule_merged[-1] = [rule_merged[-1][0], rule_merged[-1][1],str(rule_merged[-1][2].split('_')[0]) + '_' + str(rule[i + 1][2].split('_')[1]), rule[i + 1][0], rule[i + 1][1]]
                    # rule_merged.pop()
                    # rule_merged.append(tmp)
                    #rule_merged[-1][2] = str(rule_merged[-1][2].split('_')[0]) + '_' + str(rule[i + 1][2].split('_')[1])
                else:
                    #再修改movement
                    # print("movement修正：", float(rule_merged[-1][0]))
                    index_start = plot_current_data.get_index_at_t_v2(movement[joint],float(rule_merged[-1][0]))
                    index_end = plot_current_data.get_index_at_t_v2(movement[joint],float(rule[i + 1][0]))
                    # print("movement修正：", movement[joint][0][index_start])
                    for index in range(index_start-1,index_end+1):
                        # print("movement修正：",float(rule_merged[-1][2].split('_')[0].split('-')[0]))
                        movement[joint][1][index] = [int(rule_merged[-1][2].split('_')[0].split('-')[0]),0]
                    rule_merged.pop()
        else:
            # rule_merged.append(rule[i + 1])
            rule_merged.append([rule[i + 1][0],rule[i + 1][1],rule[i + 1][2],rule[i + 1][0],rule[i + 1][1]])
    rule_merged = np.array(rule_merged)
    return rule_merged,movement

def check_update_fix(merge_t,anchor,position_from_current_angle_sampled_smoothed):
    '''
    输入是position_from_current_angle_fsm
    :param merge_t:
    :param anchor:
    :param position_from_current_angle_fsm:
    :return:
    '''
    matched_index = []
    anchor_index = 0
    for i in range(merge_t.shape[0]):
        for j in range(anchor_index,len(anchor)):

            if (merge_t[i][2][0] == str(anchor[j][3])) & (merge_t[i][2][1] == str(anchor[j][4])):
                index_end_x = plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0], float(merge_t[i][0][0]))
                index_end_y = plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[1], float(merge_t[i][0][1]))
                pos_x = position_from_current_angle_sampled_smoothed[0][1][index_end_x-2]
                pos_y = position_from_current_angle_sampled_smoothed[1][1][index_end_y-2]

                dis = contour_error_v2.cauculate_dist([pos_x,pos_y],[float(anchor[j][1]),float(anchor[j][2])])
                matched_index.append([i,j])
                print("match: ", i, j, dis)
                # print(merge_t[i][2], anchor[j][3:5])

                if dis < 100:

                    index_x = plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0], float(merge_t[i][1][0]))
                    index_y = plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[1], float(merge_t[i][1][1]))
                    if len(merge_t[i][2][0].split('_')) == 2:
                        state = [[int(merge_t[i][2][0].split('_')[0].split('-')[0]),
                                  int(merge_t[i][2][0].split('_')[0].split('-')[1])],
                                 [int(merge_t[i][2][0].split('_')[1].split('-')[0]),
                                  int(merge_t[i][2][0].split('_')[1].split('-')[1])]]
                        print("state: ",state)
                        print("float(merge_t[i][0][0]): ",float(merge_t[i][0][0]))
                        position_from_current_angle_sampled_smoothed[0] = plot_current_data.position_fix_v3(position_from_current_angle_sampled_smoothed, 0, float(merge_t[i][0][0]), state, train.re_F500xy,train.re_F500xy_negative, train.r2f_F500xy, train.f2r_F500xy)
                        # position_from_current_angle_sampled_smoothed[0][1][index_x-2:] = position_from_current_angle_sampled_smoothed[0][1][index_x-2:] + (float(anchor[j][1]) - position_from_current_angle_sampled_smoothed[0][1][index_x-2])
                        # position_from_current_angle_sampled_smoothed[1][1][index_y - 2:] = position_from_current_angle_sampled_smoothed[1][1][index_y - 2:] + (float(anchor[j][2]) - position_from_current_angle_sampled_smoothed[1][1][index_y - 2])
                        position_from_current_angle_sampled_smoothed[0][1][index_x - 3:] = position_from_current_angle_sampled_smoothed[0][1][index_x - 3:] + (float(anchor[j][1]) - position_from_current_angle_sampled_smoothed[0][1][index_x - 3])
                        position_from_current_angle_sampled_smoothed[1][1][index_y - 3:] = position_from_current_angle_sampled_smoothed[1][1][index_y - 3:] + (float(anchor[j][2]) - position_from_current_angle_sampled_smoothed[1][1][index_y - 3])

                    if len(merge_t[i][2][1].split('_')) == 2:
                        state = [[int(merge_t[i][2][1].split('_')[0].split('-')[0]),
                                  int(merge_t[i][2][1].split('_')[0].split('-')[1])],
                                 [int(merge_t[i][2][1].split('_')[1].split('-')[0]),
                                  int(merge_t[i][2][1].split('_')[1].split('-')[1])]]
                        position_from_current_angle_sampled_smoothed[1] = plot_current_data.position_fix_v3(position_from_current_angle_sampled_smoothed, 1, float(merge_t[i][0][1]), state, train.re_F500xy,train.re_F500xy_negative, train.r2f_F500xy, train.f2r_F500xy)
                        # position_from_current_angle_sampled_smoothed[1][1][index_y-2:] = position_from_current_angle_sampled_smoothed[1][1][index_y-2:] + (float(anchor[j][2]) - position_from_current_angle_sampled_smoothed[1][1][index_y-2])
                        # position_from_current_angle_sampled_smoothed[0][1][index_x - 2:] = position_from_current_angle_sampled_smoothed[0][1][index_x - 2:] + (float(anchor[j][1]) - position_from_current_angle_sampled_smoothed[0][1][index_x - 2])
                        position_from_current_angle_sampled_smoothed[1][1][index_y - 3:] = position_from_current_angle_sampled_smoothed[1][1][index_y - 3:] + (float(anchor[j][2]) - position_from_current_angle_sampled_smoothed[1][1][index_y - 3])
                        position_from_current_angle_sampled_smoothed[0][1][index_x - 3:] = position_from_current_angle_sampled_smoothed[0][1][index_x - 3:] + (float(anchor[j][1]) - position_from_current_angle_sampled_smoothed[0][1][index_x - 3])

                    anchor_index = j + 1
                    break
    return position_from_current_angle_sampled_smoothed, np.array(matched_index)

def is_match_anchor(iterm1,iterm2):
    if len(iterm1.split('_')) == 2:
        if len(iterm2.split('_')) == 2:
            if (iterm1.split('_')[0].split('-')[0] == iterm2.split('_')[0].split('-')[0]) & (iterm1.split('_')[1].split('-')[0] == iterm2.split('_')[1].split('-')[0]):
                return 1
    else:
        if len(iterm2.split('_')) != 2:
            if float(iterm1) == float(iterm2):
                return 1
    return 0

def check_update_fix_v2(merge_t,anchor,position_from_current_angle_sampled_smoothed):
    '''
    输入是position_from_current_angle_fsm
    :param merge_t:
    :param anchor:
    :param position_from_current_angle_fsm:
    :return:
    '''
    matched_index = []
    anchor_index = 0
    for i in range(merge_t.shape[0]):
        for j in range(anchor_index,len(anchor)):

            # if (merge_t[i][2][0] == str(anchor[j][3])) & (merge_t[i][2][1] == str(anchor[j][4])):
            if (is_match_anchor(merge_t[i][2][0],str(anchor[j][3])) == 1) & (is_match_anchor(merge_t[i][2][1], str(anchor[j][4])) == 1):
                index_end_x = plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0], float(merge_t[i][0][0]))
                index_end_y = plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[1], float(merge_t[i][0][1]))
                pos_x = position_from_current_angle_sampled_smoothed[0][1][index_end_x-2]
                pos_y = position_from_current_angle_sampled_smoothed[1][1][index_end_y-2]

                dis = contour_error_v2.cauculate_dist([pos_x,pos_y],[float(anchor[j][1]),float(anchor[j][2])])
                matched_index.append([i,j])
                print("match: ", i, j, dis)
                # print(merge_t[i][2], anchor[j][3:5])

                if dis < 100:

                    index_x = plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0], float(merge_t[i][0][0]))
                    index_y = plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[1], float(merge_t[i][0][1]))
                    if len(merge_t[i][2][0].split('_')) == 2:
                        state = [[int(merge_t[i][2][0].split('_')[0].split('-')[0]),
                                  int(merge_t[i][2][0].split('_')[0].split('-')[1])],
                                 [int(merge_t[i][2][0].split('_')[1].split('-')[0]),
                                  int(merge_t[i][2][0].split('_')[1].split('-')[1])]]
                        position_from_current_angle_sampled_smoothed[0] = plot_current_data.position_fix_v3(position_from_current_angle_sampled_smoothed, 0, float(merge_t[i][0][0]), state, train.re_F500xy,train.re_F500xy_negative, train.r2f_F500xy, train.f2r_F500xy)
                        position_from_current_angle_sampled_smoothed[0][1][index_x - 3:] = position_from_current_angle_sampled_smoothed[0][1][index_x - 3:] + (float(anchor[j][1]) - position_from_current_angle_sampled_smoothed[0][1][index_x - 3])
                        # position_from_current_angle_sampled_smoothed[1][1][index_y - 3:] = position_from_current_angle_sampled_smoothed[1][1][index_y - 3:] + (float(anchor[j][2]) - position_from_current_angle_sampled_smoothed[1][1][index_y - 3])
                        if ((state[0][0] == 2) & (state[1][0] == 3)) | ((state[0][0] == 3) & (state[1][0] == 2)):
                            tmp_index_x = plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0], float(merge_t[i][1][0]))
                            position_from_current_angle_sampled_smoothed[0][1][tmp_index_x:] = position_from_current_angle_sampled_smoothed[0][1][tmp_index_x:] + (position_from_current_angle_sampled_smoothed[0][1][index_x - 3] - position_from_current_angle_sampled_smoothed[0][1][tmp_index_x])
                            for k in range(index_x,tmp_index_x):
                                position_from_current_angle_sampled_smoothed[0][1][k] = position_from_current_angle_sampled_smoothed[0][1][index_x - 3]
                            # print("kkkkkkkkkkkkkkkkkkkkkkkkkkkk")
                        elif (state[1][0] == 0) | (state[1][0] == 1):
                            if i < merge_t.shape[0] - 1:
                                tmp_index_x = plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0], float(merge_t[i+1][0][0]))
                            else:
                                tmp_index_x = position_from_current_angle_sampled_smoothed.shape[2]
                            for k in range(index_x,tmp_index_x):
                                position_from_current_angle_sampled_smoothed[0][1][k] = position_from_current_angle_sampled_smoothed[0][1][index_x - 3]

                    if len(merge_t[i][2][1].split('_')) == 2:
                        state = [[int(merge_t[i][2][1].split('_')[0].split('-')[0]),
                                  int(merge_t[i][2][1].split('_')[0].split('-')[1])],
                                 [int(merge_t[i][2][1].split('_')[1].split('-')[0]),
                                  int(merge_t[i][2][1].split('_')[1].split('-')[1])]]
                        position_from_current_angle_sampled_smoothed[1] = plot_current_data.position_fix_v3(position_from_current_angle_sampled_smoothed, 1, float(merge_t[i][0][1]), state, train.re_F500xy,train.re_F500xy_negative, train.r2f_F500xy, train.f2r_F500xy)
                        position_from_current_angle_sampled_smoothed[1][1][index_y - 3:] = position_from_current_angle_sampled_smoothed[1][1][index_y - 3:] + (float(anchor[j][2]) - position_from_current_angle_sampled_smoothed[1][1][index_y - 3])
                        # position_from_current_angle_sampled_smoothed[0][1][index_x - 3:] = position_from_current_angle_sampled_smoothed[0][1][index_x - 3:] + (float(anchor[j][1]) - position_from_current_angle_sampled_smoothed[0][1][index_x - 3])
                        if ((state[0][0] == 2) & (state[1][0] == 3)) | ((state[0][0] == 3) & (state[1][0] == 2)):
                            tmp_index_y = plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[1], float(merge_t[i][1][1]))
                            position_from_current_angle_sampled_smoothed[1][1][tmp_index_y:] = position_from_current_angle_sampled_smoothed[1][1][tmp_index_y:] + (position_from_current_angle_sampled_smoothed[1][1][index_y - 3] - position_from_current_angle_sampled_smoothed[1][1][tmp_index_y])
                            for k in range(index_y,tmp_index_y):
                                position_from_current_angle_sampled_smoothed[1][1][k] = position_from_current_angle_sampled_smoothed[1][1][index_y - 3]
                            # print("kkkkkkkkkkkkkkkkkkkkkkkkkkkk")
                        elif (state[1][0] == 0) | (state[1][0] == 1):
                            if i < merge_t.shape[0] - 1:
                                tmp_index_y = plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[1], float(merge_t[i+1][0][1]))
                            else:
                                tmp_index_y = position_from_current_angle_sampled_smoothed.shape[2]
                            for k in range(index_y,tmp_index_y):
                                position_from_current_angle_sampled_smoothed[1][1][k] = position_from_current_angle_sampled_smoothed[1][1][index_y - 3]

                    anchor_index = j + 1
                    break
    return position_from_current_angle_sampled_smoothed, np.array(matched_index)

def load_white_point(white_index,white_point):
    '''
    加载白名单中的第i、j、k次运动轨迹，white_index = [i,j,k]
    :param white_index:
    :param white_point:
    :return:
    '''
    if len(white_index) == 0:
        return -1
    re = white_point[white_index[0]]
    for i in range(len(white_index)-1):
        re = np.concatenate((re,white_point[white_index[i+1]]),axis=2)
    re = re[0:2,1,:]
    re = re.transpose(1,0)
    print(re.shape)
    kdtree = KDTree(re)
    return kdtree

def scatter_xy(file_name,x_angle_pair_file,y_angle_pair_file,start_point,init_state,sliding_window,gain_factor):
    '''
    :param file_name:
    :param x_angle_pair_file:
    :param y_angle_pair_file:
    :param start_point:
    :param init_state:
    :param sliding_window:
    :param gain_factor:
    :return:
    '''

    #anchor: 时间，x位置，y位置，x状态，y状态
    #arc
    # anchor = np.array([['0.0', '0', '0', '0-0_2-0', '0-0_2-0'],
    #                    ['0.0', '5.15', '5.15', '2.0', '2-0_3-0'],
    #                    ['0.0', '10.3', '0', '2-0_0-0', '3-0_1-0']])

    #line
    # anchor = np.array([['0.0', '0', '-1.5', '1-0_3-0', '1-0_3-0'],
    #                    ['0.0', '-11', '-11', '3-0_1-0', '3-0_1-0']])

    #fullarc
    anchor = np.array([['0.0', '0.0', '0.0', '0-0_2-0', '0-0_2-0'],
                       ['0.0', '5.0', '5.0', '2-0_3-0', '2.0'],
                       ['0.0', '0.0', '10.0', '3.0', '2-0_3-0'],
                       ['0.0', '-5.0', '5.0', '3-0_2-0', '3.0'],
                       ['0.0', '0.0', '0.0', '2-0_0-0', '3-0_1-0']])

    #poly
    # anchor = np.array([['0.0', '0.0', '0.0', '0-0_2-0', '0-0_2-0'],
    #                    ['0.0', '10.0', '10.0', '2-0_3-0', '2.0'],
    #                    ['0.0', '0.0', '20.0', '3.0', '2-0_3-0'],
    #                    ['0.0', '-10.0', '10.0', '3-0_2-0', '3.0'],
    #                    ['0.0', '0.0', '0.0', '2-0_0-0', '3-0_1-0']])

    # hole
    # anchor = np.array([['0.0', '0.0', '0.0', '0-0_2-0', '0.0'],
    #                    ['0.0', '5.0', '0.0', '2-0_0-0', '0-0_2-0'],
    #                    ['0.0', '5.0', '5', '0-0_3-0', '2.0'],
    #                    ['0.0', '0.0', '10', '3-0_2-0', '2.0'],
    #                    ['0.0', '5.0', '15', '2.0', '2-0_3-0'],
    #                    ['0.0', '10.0', '10', '2-0_3-0', '3.0'],
    #                    ['0.0', '5.0', '5', '3-0_1-0', '3.0'],
    #                    ['0.0', '5.0', '0', '1-0_2-0', '3-0_1-0'],
    #                    ['0.0', '10.0', '0.0', '2-0_0-0', '1.0']])


    x_ini, y_ini, z_ini = 0, 0, 0
    # instruction = [['straight', '5', '0', '0'], ['straight', '5', '5', '0'],['arc','5','5','0','5','10',-1], ['straight', '5', '0', '0'],['straight', '10', '0', '0']]  #
    # instruction = [['straight', '10', '10', '0'],['straight', '0', '20', '0'],['straight', '-10', '10', '0'],['straight', '0', '0', '0']]  #
    # instruction = [['straight', '-11', '-11', '0']]  #
    instruction = [['arc','0','0','0','0','5',-1]]#
    # instruction = [['arc', '10.3', '0', '0', '5.15', '0', -1]]  #

    point_collection = np.array([[x_ini, y_ini]])
    print("point_collection.shape: ", point_collection.shape)

    for i in instruction:
        if i[0] == 'straight':
            line_points = gcode_interpreter.line_interpolate(x_ini, y_ini, z_ini, float(i[1]), float(i[2]), float(i[3]))
            x_ini, y_ini, z_ini = float(i[1]), float(i[2]), float(i[3])
            if len(line_points) == 0:
                continue
            # print(line_points.shape)
            point_collection = np.concatenate((point_collection, line_points), axis=0)
        elif i[0] == 'arc':
            arc_points = gcode_interpreter.arc_interpolation(x_ini, y_ini, z_ini, float(i[1]), float(i[2]), float(i[3]),
                                                             float(i[4]), float(i[5]), plane, float(i[6]))
            x_ini, y_ini, z_ini = float(i[1]), float(i[2]), float(i[3])
            if len(arc_points) == 0:
                continue
            # print(arc_points.shape)
            point_collection = np.concatenate((point_collection, arc_points), axis=0)
        # print(i)

    print("point_collection.shape: ", point_collection.shape)

    kdtree = KDTree(point_collection)

    mean_error = []
    end_dist = []

    current = plot_current_data.load_pico_uvw_csv_v5(file_path + file_name + '.csv')
    print("current.shape: ", current.shape)

    current_angle_original = plot_current_data.calculate_current_angle(current, num_joints)
    current_angle_original = plot_current_data.clark_angle_to_2pi(current_angle_original, num_joints)

    print("current_angle_original.shape: ",current_angle_original.shape)

    position_from_current_angle_unmap = plot_current_data.angle_to_position_v4(current_angle_original, num_joints, pole_pair,screw_lead, start_point)
    position_from_current_angle_unmap = plot_current_data.position_negate(position_from_current_angle_unmap, num_joints)
    position_from_current_angle_unmap = plot_current_data.downsample_v2(position_from_current_angle_unmap, num_joints, 1)
    # print("position_from_current_angle_unmap.shape: ", position_from_current_angle_unmap.shape)

    coefficients_x = train.load_angle_pair_model(model_file_path, x_angle_pair_file)
    coefficients_y = train.load_angle_pair_model(model_file_path, y_angle_pair_file)
    current_angle = []
    current_angle.append(train.fit_current_angle(current_angle_original[0], coefficients_x))
    current_angle.append(train.fit_current_angle(current_angle_original[1], coefficients_y))
    current_angle = np.array(current_angle)
    print("current_angle.shape: ", current_angle.shape)


    # 绘制overview中电流和电流矢量角角的图
    fig, ax = plt.subplots(figsize=(3, 3))
    # ax.set_aspect("equal")
    # 坐标轴粗细
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # 设置坐标轴刻度数量
    ax.locator_params(nbins=3)
    # 坐标轴刻度字体大小
    # plt.xticks(fontsize=15,fontweight='bold')
    # plt.yticks(fontsize=15,fontweight='bold')
    # 设置坐标轴标签
    # ax.set_xlabel('Time (ms)',fontsize=15,fontweight='bold')
    # ax.set_ylabel('X-axis (mm)',fontsize=15,fontweight='bold')
    # 隐藏坐标轴刻度
    # ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)
    # 隐藏刻度字体
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    # ax.plot(current_angle_original[0][0][int(0.25*current_angle_original.shape[2]):int(0.5*current_angle_original.shape[2])], current_angle_original[0][1][int(0.25*current_angle_original.shape[2]):int(0.5*current_angle_original.shape[2])], 'black', linewidth=1)
    ax.plot(current_angle[0][0][int(0.25 * current_angle.shape[2]):int(0.5 * current_angle.shape[2])],current_angle[0][1][int(0.25 * current_angle.shape[2]):int(0.5 * current_angle.shape[2])], 'black', linewidth=1)
    # ax.plot(current[0][0][0][int(0.25 * current.shape[3]):int(0.5 * current.shape[3])],current[0][0][1][int(0.25 * current.shape[3]):int(0.5 * current.shape[3])], 'black', linewidth=0.3)
    # ax.plot(current[0][1][0][int(0.25 * current.shape[3]):int(0.5 * current.shape[3])],current[0][1][1][int(0.25 * current.shape[3]):int(0.5 * current.shape[3])], 'black', linewidth=0.3)
    # ax.plot(current[0][2][0][int(0.25 * current.shape[3]):int(0.5 * current.shape[3])],current[0][2][1][int(0.25 * current.shape[3]):int(0.5 * current.shape[3])], 'black',linewidth=0.3)
    # ax.scatter(current_angle_original[0][0][::100], current_angle_original[0][1][::100], color = 'black', s = 6)
    # ax.plot([2,8],[2,8],'black')
    # 显示图形
    plt.show()

    position_from_current_angle = plot_current_data.angle_to_position_v4(current_angle,num_joints,pole_pair,screw_lead,start_point)
    position_from_current_angle = plot_current_data.position_negate(position_from_current_angle,num_joints)
    # print("position_from_current_angle.shape: ",position_from_current_angle.shape)

    position_from_current_angle_sampled = plot_current_data.downsample_v2(position_from_current_angle, num_joints, 1)

    # print("position_from_current_angle_sampled.shape: ", position_from_current_angle_sampled.shape)

    position_from_current_angle_sampled_smoothed = np.array(
        [[position_from_current_angle_sampled[0][0], plot_current_data.moving_average(position_from_current_angle_sampled[0][1], sliding_window)],
         [position_from_current_angle_sampled[1][0],plot_current_data.moving_average(position_from_current_angle_sampled[1][1], sliding_window)]])

    position_from_current_angle_sampled_smoothed = position_from_current_angle_sampled_smoothed[:,:,sliding_window:-1 * sliding_window]

    # print("position_from_current_angle_sampled_smoothed.shape: ", position_from_current_angle_sampled_smoothed.shape)

    #求轨迹的差分（相当于每一时刻的速度），为了判断静止还是运动
    diff_x = np.diff(position_from_current_angle_sampled_smoothed[0][1]) * gain_factor
    diff_y = np.diff(position_from_current_angle_sampled_smoothed[1][1]) * gain_factor

    diff_1 = np.array([[position_from_current_angle_sampled_smoothed[0][0][:-1], diff_x],
                       [position_from_current_angle_sampled_smoothed[1][0][:-1], diff_y]])


    start_proportion = 0
    end_proportion = 1
    plt2 = plot_current_data.create_figure((20, 5), 'time', 'diff')
    plt2 = plot_current_data.plot_series_with_time(plt2, [diff_1[0][0], [0 for i in range(diff_1.shape[2])]], "base")
    plt2 = plot_current_data.scatter_series_with_time(plt2, diff_1[0][:, int(start_proportion * diff_1.shape[2]):int(
        end_proportion * diff_1.shape[2])], "diff_x")
    plt2 = plot_current_data.scatter_series_with_time(plt2, diff_1[1][:, int(start_proportion * diff_1.shape[2]):int(
        end_proportion * diff_1.shape[2])], "diff_y")

    print('x状态转移')
    x_movement = plot_current_data.judge_movement_test(diff_1[0], init_state[0], 100, 0.02)#20,0.02
    print("x_movement.shape: ", x_movement.shape)
    # print(x_movement)
    t_start = 0
    t_end = np.inf
    for i in range(x_movement.shape[1]):
        if (x_movement[1][i][0] == 2) | (x_movement[1][i][0] == 3):
            t_start = x_movement[0][i]
            break
    for i in range(x_movement.shape[1]):
        if (x_movement[1][x_movement.shape[1] - 1 - i][0] == 2) | (x_movement[1][x_movement.shape[1] - 1 - i][0] == 3):
            t_end = x_movement[0][x_movement.shape[1] - 1 - i]
            break
    print('y状态转移')
    y_movement = plot_current_data.judge_movement_test(diff_1[1], init_state[1], 100, 0.02)#20,0.02
    print("y_movement.shape: ",y_movement.shape)
    for i in range(y_movement.shape[1]):
        if (y_movement[1][i][0] == 2) | (y_movement[1][i][0] == 3):
            if t_start > y_movement[0][i]:
                t_start = y_movement[0][i]
            break
    for i in range(y_movement.shape[1]):
        if (y_movement[1][y_movement.shape[1] - 1 - i][0] == 2) | (y_movement[1][y_movement.shape[1] - 1 - i][0] == 3):
            if t_end < y_movement[0][y_movement.shape[1] - 1 - i]:
                t_end = y_movement[0][y_movement.shape[1] - 1 - i]
                break

    # print("起始时刻：",t_start)
    # print(plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0],t_start))
    # print("终止时刻：",t_end)
    # print(plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0],t_end))

    movement = np.array([x_movement,y_movement])
    print("movement.shape: ",movement.shape)

    # print(movement[1][:10])

    movement_x,rule_x = judge_movement_current_movement_v2(movement[0], position_from_current_angle_sampled_smoothed[0],init_state[0][0],1)
    movement_y,rule_y = judge_movement_current_movement_v2(movement[1], position_from_current_angle_sampled_smoothed[1],init_state[1][0],1)
    print("rule_x.shape: ",rule_x.shape)
    print("rule_y.shape: ",rule_y.shape)
    # index = plot_current_data.get_index_at_t_v2(movement_x,float(rule_x[1][0]))
    # print(index)
    # print(movement_x[1][index - 2])
    # print(movement_x[1][index-1])
    # print(movement_x[1][index])
    # print(movement_x[1][index-5:index+5])
    print(rule_x)
    print(rule_y)

    # rule_x_merged = merge_state_4_v2(rule_x)
    # rule_y_merged = merge_state_4_v2(rule_y)
    rule_x_merged, movement = merge_state_4_v3(rule_x, movement, 0)
    rule_y_merged, movement = merge_state_4_v3(rule_y, movement, 1)
    print("rule_x_merged.shape: ",rule_x_merged.shape)
    print("rule_y_merged.shape: ",rule_y_merged.shape)
    # print("rule_x_merged")
    # print(rule_x_merged)
    # print("rule_y_merged")
    # print(rule_y_merged)

    merge_t = merge_axis(rule_x_merged,rule_y_merged,movement)
    print("merge_t.shape: ",merge_t.shape)
    print(merge_t)

    position_from_current_angle_sampled_smoothed = position_from_current_angle_sampled_smoothed[:, :, 100:-101]

    position_from_current_angle_smoothed = copy.deepcopy(position_from_current_angle_sampled_smoothed)

    # position_from_current_angle_sampled_smoothed = plot_current_data.position_fix_v4(position_from_current_angle_sampled_smoothed, movement, num_joints)
    # print("position_from_current_angle_sampled_smoothed.shape: ", position_from_current_angle_sampled_smoothed.shape)

    # position_from_current_angle_fsm,matched_index = check_update_fix(merge_t, anchor, position_from_current_angle_sampled_smoothed)
    position_from_current_angle_fsm, matched_index = check_update_fix_v2(merge_t, anchor,position_from_current_angle_sampled_smoothed)
    print("matched_index: ")
    print(matched_index)

    # position_from_current_angle_fsm = plot_current_data.position_fix_v2(position_from_current_angle_sampled_smoothed,
    #                                                                     movement, num_joints, 100, start_point,
    #                                                                     train.re_F500xy, train.re_F500xy_negative,
    #                                                                     train.r2f_F500xy, train.f2r_F500xy)
    # print("position_from_current_angle_fsm.shape: ",position_from_current_angle_fsm.shape)
    # position_from_current_angle_fsm, matched_index = check_update(merge_t, anchor,position_from_current_angle_fsm)

    # position_from_current_angle_fsm = plot_current_data.position_fix_v4(position_from_current_angle_fsm, movement, num_joints)
    print("position_from_current_angle_fsm.shape: ", position_from_current_angle_fsm.shape)

    # print("matched_index.shape: ",matched_index.shape)
    # for i in matched_index:
    #     print(i)

    plt.figure()
    # plt.scatter(dist_position_from_current_angle_unmap[0], dist_position_from_current_angle_unmap[1], label='unmap')
    # plt.scatter(dist_position_from_current_angle_sampled_smoothed[0],
    #             dist_position_from_current_angle_sampled_smoothed[1], label='smoothed')
    plt.scatter(position_from_current_angle_fsm[0][0], position_from_current_angle_fsm[0][1], label='fsm_x')
    plt.scatter(position_from_current_angle_fsm[1][0], position_from_current_angle_fsm[1][1], label='fsm_y')
    plt.legend()
    plt.show()
    plt2.show()

    # position_from_current_angle_fsm = plot_current_data.position_fix_v2(position_from_current_angle_sampled_smoothed, movement, num_joints, 100, start_point,train.re_F500xy,train.re_F500xy_negative,train.r2f_F500xy,train.f2r_F500xy)

    position_from_current_angle_unmap = position_from_current_angle_unmap[:,:,plot_current_data.get_index_at_t_v2(position_from_current_angle_unmap[0],t_start):plot_current_data.get_index_at_t_v2(position_from_current_angle_unmap[0],t_end)]
    position_from_current_angle_smoothed = position_from_current_angle_smoothed[:,:,plot_current_data.get_index_at_t_v2(position_from_current_angle_smoothed[0],t_start):plot_current_data.get_index_at_t_v2(position_from_current_angle_smoothed[0],t_end)]

    position_from_current_angle_fsm = position_from_current_angle_fsm[:,:,plot_current_data.get_index_at_t_v2(position_from_current_angle_fsm[0],t_start):plot_current_data.get_index_at_t_v2(position_from_current_angle_fsm[0],t_end)]
    print("position_from_current_angle_fsm.shape: ", position_from_current_angle_fsm.shape)
    # plt.figure(figsize=(10,10))
    # plt.plot(position_from_current_angle_fsm[0][1],position_from_current_angle_fsm[1][1])
    # plt.show()

    #所有点ketree
    dist_position_from_current_angle_unmap = []
    for i in range(position_from_current_angle_unmap.shape[2]):
        target_point = position_from_current_angle_unmap[:, 1, i]
        # 在加载后的KD-tree中查找距离目标点最近的点
        distance, index = kdtree.query(target_point)
        # nearest_point = point_collection[index]
        dist_position_from_current_angle_unmap.append(distance)

    dist_position_from_current_angle_unmap = np.array(
        [position_from_current_angle_unmap[0][0], dist_position_from_current_angle_unmap])
    print("dist_position_from_current_angle_unmap.shape: ",dist_position_from_current_angle_unmap.shape)
    print(np.mean(dist_position_from_current_angle_unmap[1]))

    end_error_unmap = contour_error_v2.cauculate_dist(position_from_current_angle_unmap[:, 1, -1],[float(instruction[-1][1]),float(instruction[-1][2])])

    dist_position_from_current_angle_sampled_smoothed = []
    for i in range(position_from_current_angle_smoothed.shape[2]):
        target_point = position_from_current_angle_smoothed[:, 1, i]
        # 在加载后的KD-tree中查找距离目标点最近的点
        distance, index = kdtree.query(target_point)
        # nearest_point = point_collection[index]
        dist_position_from_current_angle_sampled_smoothed.append(distance)
        # print(f"目标点：{target_point}")
        # print(f"最近的点：{nearest_point}")
        # print(f"距离：{distance}")
    dist_position_from_current_angle_sampled_smoothed = np.array(
        [position_from_current_angle_smoothed[0][0], dist_position_from_current_angle_sampled_smoothed])
    print("dist_position_from_current_angle_sampled_smoothed.shape: ", dist_position_from_current_angle_sampled_smoothed.shape)
    print(np.mean(dist_position_from_current_angle_sampled_smoothed[1]))

    end_error_sampled_smoothed = contour_error_v2.cauculate_dist(position_from_current_angle_smoothed[:, 1, -1],[float(instruction[-1][1]), float(instruction[-1][2])])

    dist_position_from_current_angle_fsm = []
    for i in range(position_from_current_angle_fsm.shape[2]):
        target_point = position_from_current_angle_fsm[:, 1, i]
        # 在加载后的KD-tree中查找距离目标点最近的点
        distance, index = kdtree.query(target_point)
        # nearest_point = point_collection[index]
        dist_position_from_current_angle_fsm.append(distance)
        # print(f"目标点：{target_point}")
        # print(f"最近的点：{nearest_point}")
        # print(f"距离：{distance}")
    dist_position_from_current_angle_fsm = np.array(
        [position_from_current_angle_fsm[0][0], dist_position_from_current_angle_fsm],dtype=float)
    print("dist_position_from_current_angle_fsm.shape: ",dist_position_from_current_angle_fsm.shape)
    print(np.mean(dist_position_from_current_angle_fsm[1]))

    end_error_fsm = contour_error_v2.cauculate_dist(position_from_current_angle_fsm[:, 1, -1],[float(instruction[-1][1]), float(instruction[-1][2])])

    mean_error.append([np.mean(dist_position_from_current_angle_unmap[1]),np.mean(dist_position_from_current_angle_sampled_smoothed[1]),np.mean(dist_position_from_current_angle_fsm[1])])
    end_dist.append([end_error_unmap, end_error_sampled_smoothed, end_error_fsm])

    dist_position_from_current_angle_fsm_trans = copy.deepcopy(dist_position_from_current_angle_fsm)


    plt.figure()
    plt.plot(point_collection[:, 0], point_collection[:, 1], 'black', linewidth=1.5)#, linestyle='--'
    plt.scatter(position_from_current_angle_unmap[0][1], position_from_current_angle_unmap[1][1],
                c=dist_position_from_current_angle_unmap[1], cmap='RdYlGn', label='unmap',s = 1, vmin = 0, vmax =1)
    plt.scatter(position_from_current_angle_smoothed[0][1], position_from_current_angle_smoothed[1][1],
                c=dist_position_from_current_angle_sampled_smoothed[1], cmap='RdYlGn', label='smoothed',s = 1, vmin = 0, vmax =1)
    plt.scatter(position_from_current_angle_fsm[0][1], position_from_current_angle_fsm[1][1],
                c=dist_position_from_current_angle_fsm[1], cmap='RdYlGn', label='fsm', s=1, vmin = 0, vmax =1)
    cbar = plt.colorbar()
    cbar.set_label('contour error')
    plt.legend()
    plt.show()

    plt.figure()
    plt.scatter(dist_position_from_current_angle_unmap[0], dist_position_from_current_angle_unmap[1], label='unmap')
    plt.scatter(dist_position_from_current_angle_sampled_smoothed[0], dist_position_from_current_angle_sampled_smoothed[1], label='smoothed')
    plt.scatter(dist_position_from_current_angle_fsm[0], dist_position_from_current_angle_fsm[1], label='fsm')
    plt.legend()
    plt.show()

    interval = 1
    fig, ax = plt.subplots()
    ax.axis([-12, 12, -2, 22])#poly
    # ax.axis([-7.5, 7.5, -2.5, 12.5])#cycle
    # ax.axis([-13, 2, -13, 0.5])#line
    # ax.axis([-2, 12.3, -2, 7.15])#arc
    # ax.axis([-2, 12.3, -2, 15])
    ax.set_aspect("equal")
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 坐标轴粗细
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # 坐标轴刻度字体大小
    plt.xticks(fontsize=25,fontweight='bold')
    plt.yticks(fontsize=25,fontweight='bold')
    # 移除左边框和下边框的刻度
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # 设置坐标轴标签
    ax.set_xlabel('X-axis (mm)', fontsize=25,fontweight='bold')
    ax.set_ylabel('Y-axis (mm)', fontsize=25,fontweight='bold')
    ax.plot(point_collection[:, 0], point_collection[:, 1], 'black', linewidth=1.5,label='expected tarj')
    # ax.scatter(position_from_current_angle_fsm[0][1][::interval], position_from_current_angle_fsm[1][1][::interval],
    #             c=dist_position_from_current_angle_fsm[1][::interval], cmap='RdYlGn', label='reconstructed tarj', s=3, vmin = 0, vmax =1)#s=3
    # for index in range(len(position_from_current_angle_fsm[0][1][::interval]) - 1):
    #     ax.plot(position_from_current_angle_fsm[0][1][::interval][index:index+2],position_from_current_angle_fsm[1][1][::interval][index:index+2], linewidth=1.5)
    color = dist_position_from_current_angle_fsm[1][::interval][:-1]
    # normalizedcolor = (color - np.min(color)) / (np.max(color) - np.min(color))
    normalizedcolor = color
    segments = []
    for index in range(len(position_from_current_angle_fsm[0][1][::interval]) - 1):
        segments.append([[position_from_current_angle_fsm[0][1][::interval][index],position_from_current_angle_fsm[1][1][::interval][index]],[position_from_current_angle_fsm[0][1][::interval][index+1],position_from_current_angle_fsm[1][1][::interval][index+1]]])
    lc = LineCollection(segments, cmap='RdYlGn', norm=Normalize(0, 1),linewidth=5,label='reconstructed tarj')
    lc.set_array(normalizedcolor)
    ax.add_collection(lc)

    # # 对x和y轴设置刻度最大数量
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='RdYlGn'), ax=ax)
    cbar.set_label('contour error (mm)', fontsize=25,fontweight='bold')
    cbar.ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=25, weight='bold') #设置colorbar刻度字体大小和粗细

    # legend = plt.legend(loc=1,fontsize='large')
    # legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), shadow=False, ncol=2, fontsize='large',
    #                     fancybox=False, edgecolor='none')  #
    # # 添加图例，并设置字体粗细
    # for text in legend.get_texts():
    #     text.set_fontweight('bold')

    fig, ax = plt.subplots()
    ax.axis([-12, 12, -2, 22])#poly
    # ax.axis([-7.5, 7.5, -2.5, 12.5])#cycle
    # ax.axis([-13, 2, -13, 0.5])#line
    # ax.axis([-2, 12.3, -2, 7.15])#arc
    # ax.axis([-2, 12.3, -2, 15])
    ax.set_aspect("equal")
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 坐标轴粗细
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # 坐标轴刻度字体大小
    plt.xticks(fontsize=25,fontweight='bold')
    plt.yticks(fontsize=25,fontweight='bold')
    # 移除左边框和下边框的刻度
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # 设置坐标轴标签
    ax.set_xlabel('X-axis (mm)', fontsize=25,fontweight='bold')
    ax.set_ylabel('Y-axis (mm)', fontsize=25,fontweight='bold')
    ax.plot(point_collection[:, 0], point_collection[:, 1], 'black', linewidth=1.5,label='expected tarj')
    # ax.scatter(position_from_current_angle_smoothed[0][1][::interval], position_from_current_angle_smoothed[1][1][::interval],
    #             c=dist_position_from_current_angle_sampled_smoothed[1][::interval], cmap='RdYlGn', label='reconstructed tarj-', s=3, vmin=0,vmax=1)

    color = dist_position_from_current_angle_sampled_smoothed[1][::interval][:-1]
    # normalizedcolor = (color - np.min(color)) / (np.max(color) - np.min(color))
    normalizedcolor = color
    segments = []
    for index in range(len(position_from_current_angle_smoothed[0][1][::interval]) - 1):
        segments.append([[position_from_current_angle_smoothed[0][1][::interval][index],
                          position_from_current_angle_smoothed[1][1][::interval][index]],
                         [position_from_current_angle_smoothed[0][1][::interval][index + 1],
                          position_from_current_angle_smoothed[1][1][::interval][index + 1]]])
    lc = LineCollection(segments, cmap='RdYlGn', norm=Normalize(0, 1), linewidth=5)
    lc.set_array(normalizedcolor)
    ax.add_collection(lc)

    # # 对x和y轴设置刻度最大数量
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='RdYlGn'), ax=ax)
    cbar.set_label('contour error (mm)', fontsize=25,fontweight='bold')
    cbar.ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=25, weight='bold')  # 设置colorbar刻度字体大小。
    # plt.legend(loc=1,fontsize='small')

    fig, ax = plt.subplots()
    ax.axis([-12, 12, -2, 22])#poly
    # ax.axis([-7.5, 7.5, -2.5, 12.5])#cycle
    # ax.axis([-13, 2, -13, 0.5])#line
    # ax.axis([-2, 12.3, -2, 7.15])#arc
    # ax.axis([-2, 12.3, -2, 15])
    ax.set_aspect("equal")
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 坐标轴粗细
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # 坐标轴刻度字体大小
    plt.xticks(fontsize=25,fontweight='bold')
    plt.yticks(fontsize=25,fontweight='bold')
    # 移除左边框和下边框的刻度
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # 设置坐标轴标签
    ax.set_xlabel('X-axis (mm)', fontsize=25,fontweight='bold')
    ax.set_ylabel('Y-axis (mm)', fontsize=25,fontweight='bold')
    ax.plot(point_collection[:, 0], point_collection[:, 1], 'black', linewidth=1.5,label='expected tarj')
    # ax.scatter(position_from_current_angle_unmap[0][1][::interval], position_from_current_angle_unmap[1][1][::interval],
    #             c=dist_position_from_current_angle_unmap[1][::interval], cmap='RdYlGn', label='reconstructed tarj--',s = 3, vmin = 0, vmax =1)

    color = dist_position_from_current_angle_unmap[1][::interval][:-1]
    # normalizedcolor = (color - np.min(color)) / (np.max(color) - np.min(color))
    normalizedcolor = color
    segments = []
    for index in range(len(position_from_current_angle_unmap[0][1][::interval]) - 1):
        segments.append([[position_from_current_angle_unmap[0][1][::interval][index],
                          position_from_current_angle_unmap[1][1][::interval][index]],
                         [position_from_current_angle_unmap[0][1][::interval][index + 1],
                          position_from_current_angle_unmap[1][1][::interval][index + 1]]])
    lc = LineCollection(segments, cmap='RdYlGn', norm=Normalize(0, 1), linewidth=5)#
    lc.set_array(normalizedcolor)
    ax.add_collection(lc)

    # # 对x和y轴设置刻度最大数量
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='RdYlGn'), ax=ax)
    cbar.set_label('contour error (mm)', fontsize=25,fontweight='bold')
    cbar.ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=25, weight='bold')  # 设置colorbar刻度字体大小。
    # plt.legend(loc=1,fontsize='small')
    plt.show()

    # plt.figure()
    # # plt.scatter(dist_position_from_current_angle_unmap[0], dist_position_from_current_angle_unmap[1], label='unmap')
    # # plt.scatter(dist_position_from_current_angle_sampled_smoothed[0],
    # #             dist_position_from_current_angle_sampled_smoothed[1], label='smoothed')
    # plt.scatter(dist_position_from_current_angle_fsm_trans[0], dist_position_from_current_angle_fsm_trans[1], label='fsm')
    # plt.legend()
    # plt.show()

    # 绘制overview中还原轨迹的图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    # 设置坐标系标签的大小，标签与坐标轴的距离
    ax.set_xlabel('X-axis (mm)', size=25, labelpad=15, fontweight='bold')  #
    ax.set_ylabel('Y-axis (mm)', size=25, labelpad=15, fontweight='bold')  #
    ax.set_zlabel('Z-axis (mm)', size=25, labelpad=15, fontweight='bold')  #
    # 坐标轴粗细
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # 设置坐标系刻度的大小
    ax.tick_params(labelsize=15)
    # 设置坐标轴刻度数量
    ax.locator_params(nbins=3)
    # 坐标轴刻度字体大小
    plt.xticks(fontsize=25, fontweight='bold')
    plt.yticks(fontsize=25, fontweight='bold')
    # 设置 z 轴刻度的字体大小
    ax.tick_params(axis='z', labelsize=20)
    # 获取 Z 轴刻度对象
    z_ticks = ax.get_zaxis().get_major_ticks()
    # 设置 Z 轴刻度字体的粗细
    for tick in z_ticks:
        tick.label1.set_fontweight('bold')
    # # 设置坐标系范围
    # ax.set_xlim([-0.1, 1.1])
    # ax.set_ylim([-0.1, 1.1])
    # ax.set_zlim([-0.1, 1.1])
    # # 设置坐标系刻度
    # ax.set_xticks([0.0, 1.0])
    # ax.set_zticks([0.0, 1.0])
    # ax.set_yticks([0.0, 1.0])
    anchor = [[0, 0], [-10, 10], [0, 20], [10, 10], [0, 0]]
    anchor = np.array(anchor)
    print("anchor.shape: ", anchor.shape)
    print(np.array(anchor[:, 0], dtype=float), np.array(anchor[:, 1], dtype=float))
    ax.plot(position_from_current_angle_fsm[0][1], position_from_current_angle_fsm[1][1], [0 for i in range(len(position_from_current_angle_fsm[1][1]))], 'black', linewidth=3.5)
    #ax.scatter(np.array(anchor[:, 0], dtype=float), np.array(anchor[:, 1], dtype=float), s=80, color='black')
    ax.legend()
    plt.show()

    print('c.e.')
    for i in mean_error:
        print(i)
    print('p.e.')
    for i in end_dist:
        print(i)

    return

# scatter_xy('20230726-X0.5+-X10.5+-Y0+-Y10+-F500-50hz-0001-1690337824576-1977','20230708-X0+-X100+-F500-0001','20230616-Y0+-Y100-F500-0001',[-0.5,0], [[0,0],[0,0]], 20, 1000)
scatter_xy('20230729-X0+-Y0+-fullarc-R5-F500-50hz-0001-1690633349264-176','20230708-X0+-X100+-F500-0001','20230616-Y0+-Y100-F500-0001',[0,0], [[0,0],[0,0]], 20, 1000)




