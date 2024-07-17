import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import gcode_interpreter
import plot_opcua_data
import plot_current_data
import train
import h5py
import math
import os

screw_lead = 10
num_joints = 2
sample_time = 0.001
pole_pair = 4

file_path = './'
train_file_path = 'C:/Users/lsj\PycharmProjects/experiment0616/line/'
model_file_path =  './train_data/'


offset = [-148.55306, -295.10247]#828D
plane = 1#1为XY平面；2为YZ平面；3为XZ平面

def cauculate_dist(p1,p2):
    distance = math.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    return distance

def reconstruct(file_name,x_angle_pair_file,y_angle_pair_file,t_offfset,start_point,init_state,sliding_window,gain_factor):

    data, data_encoder = plot_opcua_data.load_csv_v2(file_path + 'DataLogger_' + file_name + '.csv', 3)


    h_only_sample_points = plot_opcua_data.interp_list_only_sample_points(data_encoder)
    # print("data_encoder.shape: ", data_encoder.shape)
    # print("h_only_sample_points.shape: ", h_only_sample_points.shape)

    h_only_sample_points[0][1] = h_only_sample_points[0][1] - offset[0]
    h_only_sample_points[1][1] = h_only_sample_points[1][1] - offset[1]

    current = plot_current_data.load_pico_uvw_csv_v5(file_path + file_name+'.csv')
    # print("current.shape: ",current.shape)

    current_angle = plot_current_data.calculate_current_angle(current,num_joints)
    current_angle = plot_current_data.clark_angle_to_2pi(current_angle,num_joints)
    # print("current_angle.shape: ",current_angle.shape)

    current_angle[0][0] = current_angle[0][0] + t_offfset[0]-t_offfset[1] - 28800000
    current_angle[1][0] = current_angle[1][0] + t_offfset[0]-t_offfset[1] - 28800000

    position_from_current_angle_unmap = plot_current_data.angle_to_position_v4(current_angle, num_joints, pole_pair,screw_lead, start_point)
    position_from_current_angle_unmap = plot_current_data.position_negate(position_from_current_angle_unmap, num_joints)
    position_from_current_angle_unmap = plot_current_data.downsample_v2(position_from_current_angle_unmap, num_joints, 1)
    # print("position_from_current_angle_unmap.shape: ", position_from_current_angle_unmap.shape)

    coefficients_x = train.load_angle_pair_model(train_file_path, x_angle_pair_file)
    coefficients_y = train.load_angle_pair_model(train_file_path, y_angle_pair_file)
    current_angle[0] = train.fit_current_angle(current_angle[0],coefficients_x)
    current_angle[1] = train.fit_current_angle(current_angle[1], coefficients_y)

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

    # print("diff_1.shape: ", diff_1.shape)

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

    print("起始时刻：",t_start)
    print(plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0],t_start))
    print("终止时刻：",t_end)
    print(plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0],t_end))

    movement = np.array([x_movement,y_movement])

    #position_from_current_angle_fsm = plot_current_data.angle_to_position_v5(current_angle, movement, 50, num_joints, pole_pair, screw_lead, start_point)
    position_from_current_angle_fsm = plot_current_data.position_fix_v2(position_from_current_angle_sampled_smoothed, movement, num_joints, 100, start_point,train.re_F500xy,train.re_F500xy_negative,train.r2f_F500xy,train.f2r_F500xy)
    # position_from_current_angle_fsm = plot_current_data.position_negate(position_from_current_angle_fsm, num_joints)
    # print("position_from_current_angle_fsm.shape: ", position_from_current_angle_fsm.shape)

    position_from_current_angle_unmap = position_from_current_angle_unmap[:,:,plot_current_data.get_index_at_t_v2(position_from_current_angle_unmap[0],t_start):plot_current_data.get_index_at_t_v2(position_from_current_angle_unmap[0],t_end)]
    position_from_current_angle_sampled_smoothed = position_from_current_angle_sampled_smoothed[:,:,plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0],t_start):plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0],t_end)]
    position_from_current_angle_fsm = position_from_current_angle_fsm[:,:,plot_current_data.get_index_at_t_v2(position_from_current_angle_fsm[0],t_start):plot_current_data.get_index_at_t_v2(position_from_current_angle_fsm[0],t_end)]

    return position_from_current_angle_unmap,position_from_current_angle_sampled_smoothed,position_from_current_angle_fsm

def reconstruct_fromh5(file_name,x_angle_pair_file,y_angle_pair_file,start_point,init_state,sliding_window,gain_factor):
    '''
    批量化评估电流还原轨迹精度，记得修改xyz_ini和instruction
    :param file_name: 存有电流的h5文件
    :param x_angle_pair_file:
    :param y_angle_pair_file:
    :param start_point:
    :param init_state:
    :param sliding_window:
    :param gain_factor:
    :return:
    '''
    x_ini, y_ini, z_ini = 0, 0, 0
    #instruction = [['straight', '10', '10', '0'],['straight', '0', '20', '0'],['straight', '-10', '10', '0'],['straight', '0', '0', '0']]  #
    instruction = [['straight', '60', '60', '0']]  #
    # instruction = [['arc','0','0','0','0','1',-1]]#

    point_collection = np.array([[0, 0]])
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

    folder_path = train_file_path + '/error/' + file_name
    if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folder_path)
    print(folder_path)

    h5f = h5py.File(folder_path + file_name + '.h5', 'w')

    for e_index in range(20):

        f = h5py.File(train_file_path + file_name + ".h5", "r")
        current_angle_original = np.array(f['current_angle_original_'+str(e_index)])
        f.close()

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

        # print("diff_1.shape: ", diff_1.shape)

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
        print(plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0],t_start))
        # print("终止时刻：",t_end)
        print(plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0],t_end))

        movement = np.array([x_movement,y_movement])

        position_from_current_angle_fsm = plot_current_data.position_fix_v2(position_from_current_angle_sampled_smoothed, movement, num_joints, 100, start_point,train.re_F500xy,train.re_F500xy_negative,train.r2f_F500xy,train.f2r_F500xy)

        position_from_current_angle_unmap = position_from_current_angle_unmap[:,:,plot_current_data.get_index_at_t_v2(position_from_current_angle_unmap[0],t_start):plot_current_data.get_index_at_t_v2(position_from_current_angle_unmap[0],t_end)]
        position_from_current_angle_sampled_smoothed = position_from_current_angle_sampled_smoothed[:,:,plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0],t_start):plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0],t_end)]
        position_from_current_angle_fsm = position_from_current_angle_fsm[:,:,plot_current_data.get_index_at_t_v2(position_from_current_angle_fsm[0],t_start):plot_current_data.get_index_at_t_v2(position_from_current_angle_fsm[0],t_end)]

        dist_position_from_current_angle_unmap = []
        for i in range(position_from_current_angle_unmap.shape[2]):
            target_point = position_from_current_angle_unmap[:, 1, i]
            # 在加载后的KD-tree中查找距离目标点最近的点
            distance, index = kdtree.query(target_point)
            # nearest_point = point_collection[index]
            dist_position_from_current_angle_unmap.append(distance)

        dist_position_from_current_angle_unmap = np.array(
            [position_from_current_angle_unmap[0][0], dist_position_from_current_angle_unmap])
        print(dist_position_from_current_angle_unmap.shape)
        print(np.mean(dist_position_from_current_angle_unmap[1]))

        end_error_unmap = cauculate_dist(position_from_current_angle_unmap[:, 1, -1],[float(instruction[-1][1]),float(instruction[-1][2])])

        dist_position_from_current_angle_sampled_smoothed = []
        for i in range(position_from_current_angle_sampled_smoothed.shape[2]):
            target_point = position_from_current_angle_sampled_smoothed[:, 1, i]
            # 在加载后的KD-tree中查找距离目标点最近的点
            distance, index = kdtree.query(target_point)
            # nearest_point = point_collection[index]
            dist_position_from_current_angle_sampled_smoothed.append(distance)
            # print(f"目标点：{target_point}")
            # print(f"最近的点：{nearest_point}")
            # print(f"距离：{distance}")
        dist_position_from_current_angle_sampled_smoothed = np.array(
            [position_from_current_angle_sampled_smoothed[0][0], dist_position_from_current_angle_sampled_smoothed])
        print(dist_position_from_current_angle_sampled_smoothed.shape)
        print(np.mean(dist_position_from_current_angle_sampled_smoothed[1]))

        end_error_sampled_smoothed = cauculate_dist(position_from_current_angle_sampled_smoothed[:, 1, -1],[float(instruction[-1][1]), float(instruction[-1][2])])

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
        print(dist_position_from_current_angle_fsm.shape)
        print(np.mean(dist_position_from_current_angle_fsm[1]))

        end_error_fsm = cauculate_dist(position_from_current_angle_fsm[:, 1, -1],[float(instruction[-1][1]), float(instruction[-1][2])])

        mean_error.append([np.mean(dist_position_from_current_angle_unmap[1]),np.mean(dist_position_from_current_angle_sampled_smoothed[1]),np.mean(dist_position_from_current_angle_fsm[1])])
        end_dist.append([end_error_unmap,end_error_sampled_smoothed,end_error_fsm])

        plt.figure()
        # plt.scatter(dist_position_from_current_angle_unmap[0], dist_position_from_current_angle_unmap[1], label='unmap')
        # plt.scatter(dist_position_from_current_angle_sampled_smoothed[0],
        #             dist_position_from_current_angle_sampled_smoothed[1], label='smoothed')
        plt.scatter(dist_position_from_current_angle_fsm[0], dist_position_from_current_angle_fsm[1], label='fsm')
        plt.legend()
        plt.savefig(folder_path+'/dist_position_from_current_angle_fsm'+str(e_index)+'.png')
        # plt.show()

        h5f.create_dataset('dist_position_from_current_angle_fsm_' + str(e_index), data=dist_position_from_current_angle_fsm)

    h5f.close()

    f = open(folder_path + '/' + 'mean_error.txt','w')
    for i in range(len(mean_error)):
        f.write(str(mean_error[i][0]) + ',' + str(mean_error[i][1]) + ',' + str(mean_error[i][2]))
        f.write('\n')
    f.close()

    f = open(folder_path + '/' + 'end_dist.txt', 'w')
    for i in range(len(end_dist)):
        f.write(str(end_dist[i][0]) + ',' + str(end_dist[i][1]) + ',' + str(end_dist[i][2]))
        f.write('\n')
    f.close()

    return

def reconstruct_fromh5_one(file_name,x_angle_pair_file,y_angle_pair_file,start_point,init_state,sliding_window,gain_factor):
    '''
    评估指定一次电流还原轨迹的精度
    :param file_name:
    :param x_angle_pair_file:
    :param y_angle_pair_file:
    :param start_point:
    :param init_state:
    :param sliding_window:
    :param gain_factor:
    :return:
    '''
    x_ini, y_ini, z_ini = 0.5, 0, 0
    instruction = [['straight', '10', '10', '0']]  #

    point_collection = np.array([[0, 0]])
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

    e_index = 4

    f = h5py.File(train_file_path + file_name + ".h5", "r")
    current_angle_original = np.array(f['current_angle_original_'+str(e_index)])
    f.close()

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

    # print("diff_1.shape: ", diff_1.shape)

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
    print(plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0],t_start))
    # print("终止时刻：",t_end)
    print(plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0],t_end))

    movement = np.array([x_movement,y_movement])

    position_from_current_angle_fsm = plot_current_data.position_fix_v2(position_from_current_angle_sampled_smoothed, movement, num_joints, 100, start_point,train.re_F500xy,train.re_F500xy_negative,train.r2f_F500xy,train.f2r_F500xy)

    # position_from_current_angle_unmap = position_from_current_angle_unmap[:,:,plot_current_data.get_index_at_t_v2(position_from_current_angle_unmap[0],t_start):plot_current_data.get_index_at_t_v2(position_from_current_angle_unmap[0],t_end)]
    # position_from_current_angle_sampled_smoothed = position_from_current_angle_sampled_smoothed[:,:,plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0],t_start):plot_current_data.get_index_at_t_v2(position_from_current_angle_sampled_smoothed[0],t_end)]
    # position_from_current_angle_fsm = position_from_current_angle_fsm[:,:,plot_current_data.get_index_at_t_v2(position_from_current_angle_fsm[0],t_start):plot_current_data.get_index_at_t_v2(position_from_current_angle_fsm[0],t_end)]

    dist_position_from_current_angle_unmap = []
    for i in range(position_from_current_angle_unmap.shape[2]):
        target_point = position_from_current_angle_unmap[:, 1, i]
        # 在加载后的KD-tree中查找距离目标点最近的点
        distance, index = kdtree.query(target_point)
        # nearest_point = point_collection[index]
        dist_position_from_current_angle_unmap.append(distance)

    dist_position_from_current_angle_unmap = np.array(
        [position_from_current_angle_unmap[0][0], dist_position_from_current_angle_unmap])
    print(dist_position_from_current_angle_unmap.shape)
    print(np.mean(dist_position_from_current_angle_unmap[1]))

    end_error_unmap = cauculate_dist(position_from_current_angle_unmap[:, 1, -1],[float(instruction[-1][1]),float(instruction[-1][2])])

    dist_position_from_current_angle_sampled_smoothed = []
    for i in range(position_from_current_angle_sampled_smoothed.shape[2]):
        target_point = position_from_current_angle_sampled_smoothed[:, 1, i]
        # 在加载后的KD-tree中查找距离目标点最近的点
        distance, index = kdtree.query(target_point)
        # nearest_point = point_collection[index]
        dist_position_from_current_angle_sampled_smoothed.append(distance)
        # print(f"目标点：{target_point}")
        # print(f"最近的点：{nearest_point}")
        # print(f"距离：{distance}")
    dist_position_from_current_angle_sampled_smoothed = np.array(
        [position_from_current_angle_sampled_smoothed[0][0], dist_position_from_current_angle_sampled_smoothed])
    print(dist_position_from_current_angle_sampled_smoothed.shape)
    print(np.mean(dist_position_from_current_angle_sampled_smoothed[1]))

    end_error_sampled_smoothed = cauculate_dist(position_from_current_angle_sampled_smoothed[:, 1, -1],[float(instruction[-1][1]), float(instruction[-1][2])])

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
        [position_from_current_angle_fsm[0][0], dist_position_from_current_angle_fsm])
    print(dist_position_from_current_angle_fsm.shape)
    print(np.mean(dist_position_from_current_angle_fsm[1]))

    end_error_fsm = cauculate_dist(position_from_current_angle_fsm[:, 1, -1],[float(instruction[-1][1]), float(instruction[-1][2])])

    mean_error.append([np.mean(dist_position_from_current_angle_unmap[1]),np.mean(dist_position_from_current_angle_sampled_smoothed[1]),np.mean(dist_position_from_current_angle_fsm[1])])
    end_dist.append([end_error_unmap,end_error_sampled_smoothed,end_error_fsm])

    folder_path = train_file_path +'/error/'+ file_name
    if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folder_path)
    print(folder_path)

    # plt.figure()
    # plt.scatter(dist_position_from_current_angle_unmap[0], dist_position_from_current_angle_unmap[1], label='unmap')
    # plt.scatter(dist_position_from_current_angle_sampled_smoothed[0],
    #             dist_position_from_current_angle_sampled_smoothed[1], label='smoothed')
    # plt.scatter(dist_position_from_current_angle_fsm[0], dist_position_from_current_angle_fsm[1], label='fsm')
    # plt.legend()
    # plt.savefig(folder_path+'/'+str(e_index)+'.png')
    # plt.show()

    start_proportion = 0
    end_proportion = 1
    plt = plot_current_data.create_figure((20, 5), 'time', 'h_only_sample_points')
    plt.title(file_name)

    plt = plot_current_data.plot_series_with_time(plt, [diff_1[0][0], [0 for i in range(diff_1.shape[2])]], "base")

    plt = plot_current_data.scatter_series_with_time(plt, diff_1[0][:, int(start_proportion * diff_1.shape[2]):int(
        end_proportion * diff_1.shape[2])], "diff_x")

    # plt = plot_current_data.scatter_series_with_time(plt, h_only_sample_points[0][:,int(start_proportion*h_only_sample_points.shape[2]):int(end_proportion*h_only_sample_points.shape[2])],"h_only_sample_points_x")

    plt = plot_current_data.plot_series_with_time(plt, position_from_current_angle_fsm[0][:,
                                                       int(start_proportion * position_from_current_angle_fsm.shape[
                                                           2]):int(
                                                           end_proportion * position_from_current_angle_fsm.shape[2])],
                                                  "position_from_current_angle_fsm_x")

    plt = plot_current_data.plot_series_with_time(plt, position_from_current_angle[0][:,
                                                       int(start_proportion * position_from_current_angle.shape[2]):int(
                                                           end_proportion * position_from_current_angle.shape[2])],
                                                  "position_from_current_angle_x")

    # plt = plot_current_data.plot_series_with_time(plt, current_angle[0][:,
    #                                                    int(start_proportion * current_angle.shape[2]):int(
    #                                                        end_proportion * current_angle.shape[2])], "current_angle_x")
    # plt = plot_current_data.plot_series_with_time(plt, current_angle[1][:,
    #                                                    int(start_proportion * current_angle.shape[2]):int(
    #                                                        end_proportion * current_angle.shape[2])], "current_angle_y")

    plt = plot_current_data.scatter_series_with_time(plt, position_from_current_angle_sampled_smoothed[0][:,
                                                          int(start_proportion *
                                                              position_from_current_angle_sampled_smoothed.shape[
                                                                  2]):int(
                                                              end_proportion *
                                                              position_from_current_angle_sampled_smoothed.shape[2])],
                                                     "position_from_current_angle_sampled_smoothed_x")

    start_proportion = 0
    end_proportion = 1
    plt2 = plot_current_data.create_figure((20, 5), 'time', 'h_only_sample_points')
    plt2.title(file_name)
    plt2 = plot_current_data.plot_series_with_time(plt2, [diff_1[0][0], [0 for i in range(diff_1.shape[2])]], "base")
    plt2 = plot_current_data.scatter_series_with_time(plt2, diff_1[1][:, int(start_proportion * diff_1.shape[2]):int(
        end_proportion * diff_1.shape[2])], "diff_y")

    plt2 = plot_current_data.plot_series_with_time(plt2, position_from_current_angle_fsm[1][:,
                                                         int(start_proportion * position_from_current_angle_fsm.shape[
                                                             2]):int(
                                                             end_proportion * position_from_current_angle_fsm.shape[
                                                                 2])], "position_from_current_angle_fsm_y")
    plt2 = plot_current_data.plot_series_with_time(plt2, position_from_current_angle[1][:,
                                                         int(start_proportion * position_from_current_angle.shape[
                                                             2]):int(
                                                             end_proportion * position_from_current_angle.shape[2])],
                                                   "position_from_current_angle_y")
    plt2 = plot_current_data.scatter_series_with_time(plt2, position_from_current_angle_sampled_smoothed[1][:,
                                                            int(start_proportion *
                                                                position_from_current_angle_sampled_smoothed.shape[
                                                                    2]):int(end_proportion *
                                                                            position_from_current_angle_sampled_smoothed.shape[
                                                                                2])],
                                                      "position_from_current_angle_sampled_smoothed_y")

    # plt = plot_current_data.plot_series_with_time(plt, ideal[:,int(start_proportion*ideal.shape[1]):int(end_proportion*ideal.shape[1])],"ideal")
    # plt.savefig('current_anchor.pdf', bbox_inches='tight')
    plt.show()
    plt2.show()

    return
