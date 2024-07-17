import numpy as np
import plot_current_data
import matplotlib.pyplot as plt
import os

import h5py

screw_lead = 10
num_joints = 2
sample_time = 0.001
pole_pair = 4
sliding_window = 50

train_file_path = './train_data/'
file_path = './train_data/'

def create_dataset_angle_pair_line(file_path, file_name, num_joints):
    current = plot_current_data.load_pico_uvw_csv_v3(file_path + file_name)
    current = np.array([current])
    print("current.shape: ", current.shape)

    current_angle = plot_current_data.calculate_current_angle(current, num_joints)
    current_angle = plot_current_data.clark_angle_to_2pi(current_angle, num_joints)
    print("current_angle.shape: ", current_angle.shape)

    position_from_current_angle = plot_current_data.angle_to_position_v3(current_angle, num_joints, pole_pair,
                                                                         screw_lead, 0)
    position_from_current_angle = plot_current_data.position_negate(position_from_current_angle, 1)
    print("position_from_current_angle.shape: ", position_from_current_angle.shape)

    # plt.figure()
    # plt.scatter(current_angle[0][0], current_angle[0][1])
    # plt.show()

    indices = []
    # 找到分段的索引
    for i in range(0, current_angle.shape[2] - 1):
        if abs(current_angle[0][1][i + 1] - current_angle[0][1][i]) > 6.2:
            indices.append(i)
    # 按照索引将数组分段
    current_angle = current_angle.transpose(2, 0, 1)
    current_angle_splited = np.split(current_angle, indices)
    print("len(current_angle_splited)",len(current_angle_splited))
    current_angle_splited = current_angle_splited[1:-2]
    print("len(current_angle_splited): ",len(current_angle_splited))

    print("current_angle_splited[0].shape: ",current_angle_splited[0].shape)

    #画图代码
    ap = []
    k = 2 * np.pi / (current_angle_splited[1][0][0][0] - current_angle_splited[1][-1][0][0])
    print("k: ", k)

    for num in range(3):
        for j in range(current_angle_splited[num].shape[0]):
            ap.append([current_angle_splited[num][j][0][0], 2 * np.pi + k * (current_angle_splited[num][j][0][0] - current_angle_splited[num][0][0][0])])
            # ap.append([current_angle_splited[1][j][0][0], (0.5 * np.pi + 2 * np.pi + k * (current_angle_splited[1][j][0][0] - current_angle_splited[1][0][0][0]))%(2 * np.pi)])

    ap = np.array(ap)

    print("ap.shape: ", ap.shape)
    ap = ap[::20]
    print("ap.shape: ", ap.shape)

    # t_ini = current_angle_splited[1][0][0][0]
    t_ini = current_angle_splited[0][0][0][0]
    for num in range(3):
        for j in range(current_angle_splited[num].shape[0]):
            current_angle_splited[num][j][0][0] = current_angle_splited[num][j][0][0] - t_ini
    ap[:,0] = ap[:,0] - t_ini

    # plt.figure(figsize=(10,5))
    # plt.xlabel("Time (ms)", fontsize=15)
    # plt.ylabel("Angle-X (rad)", fontsize=15)
    # plt.tick_params(labelsize=15)
    # for num in range(2):
    #     # plt.scatter(current_angle_splited[num][:,0,0], current_angle_splited[num][:,0,1],s = 2,color='black')
    #     plt.scatter(current_angle_splited[num][::20, 0, 0], current_angle_splited[num][::20, 0, 1], s=2, color='black')
    # plt.scatter(current_angle_splited[2][::20, 0, 0], current_angle_splited[2][::20, 0, 1], label="current_vector_angle", s=2, color='black')
    # # plt.scatter(current_angle_splited[2][:, 0, 0], current_angle_splited[2][:, 0, 1], label="current_vector_angle", s=2, color='black')
    # plt.scatter(ap[:,0], ap[:,1],label = "rotor_position_angle",s = 2,color='gray')
    # # plt.plot(current_angle_splited[1][:, 0, 0], current_angle_splited[1][:, 0, 1], label="current_phase_angle",linewidth = 2,color='black')
    # # plt.plot(ap[:, 0], ap[:, 1], label="rotor_position_angle",linewidth = 2,color='blue')#,linestyle='--'
    # plt.legend(loc="upper right")
    # plt.show()

    fig, ax = plt.subplots(figsize=(5, 2.5))
    plt.grid(True)

    # 坐标轴粗细
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # 坐标轴刻度字体大小
    plt.xticks(fontsize=15,fontweight='bold')
    plt.yticks(fontsize=15,fontweight='bold')

    plt.yticks([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi],
               ['0', '$\mathdefault{0.5\pi}$', '$\mathdefault{\pi}$', '$\mathdefault{1.5\pi}$',
                '$\mathdefault{2\pi}$'])

    # 设置坐标轴标签
    ax.set_xlabel('Time (ms)',fontsize=15,fontweight='bold')
    ax.set_ylabel('Angle-X (rad)',fontsize=15,fontweight='bold')

    for num in range(2):
        ax.scatter(current_angle_splited[num][::20, 0, 0], current_angle_splited[num][::20, 0, 1], s=12, color='black')
    ax.scatter(current_angle_splited[2][::20, 0, 0], current_angle_splited[2][::20, 0, 1],
                label="current_vector_angle", s=12, color='black')
    # ax.plot(current_angle_splited[2][::20, 0, 0], current_angle_splited[2][::20, 0, 1],
    #            label="current_vector_angle", linewidth=3, color='black')
    # plt.scatter(current_angle_splited[2][:, 0, 0], current_angle_splited[2][:, 0, 1], label="current_vector_angle", s=2, color='black')
    ax.scatter(ap[:, 0], ap[:, 1], label="rotor_position_angle", s=12, color='gray')
    # ax.plot(ap[:, 0], ap[:, 1], label="rotor_position_angle", linewidth=3, color='gray')

    plt.show()

    angle_pair = []

    # for i in range(len(current_angle_splited)):
    for i in range(3):
    # for i in range(1):
        current_angle_tmp = current_angle_splited[i].transpose(1, 2, 0)
        print("current_angle_tmp.shape: ", current_angle_tmp.shape)
        k = 2 * np.pi / (current_angle_tmp[0][0][0] - current_angle_tmp[0][0][-1])
        for j in range(current_angle_tmp.shape[2]):
            angle_pair.append([current_angle_tmp[0][1][j], 2 * np.pi + k * (current_angle_tmp[0][0][j] - current_angle_tmp[0][0][0])])
            # angle_pair.append([current_angle_tmp[0][1][j], (0.5 * np.pi + 2 * np.pi + k * (current_angle_tmp[0][0][j] - current_angle_tmp[0][0][0])) % (2 * np.pi)])
    angle_pair = np.array(angle_pair)
    print("angle_pair.shape: ",angle_pair.shape)

    plt.figure(figsize=(7,7))
    plt.xlabel("Current_vector_angle (rad)", fontsize=15)
    plt.ylabel("Rotor_position_angle (rad)", fontsize=15)
    plt.tick_params(labelsize=15)#labelsize=15
    # plt.scatter(angle_pair[:, 0], angle_pair[:, 1],s = 2, color='black')
    plt.plot(angle_pair[:, 0], angle_pair[:, 1], linewidth=2, color='black')
    # plt.xticks([0,0.5*np.pi,np.pi,1.5*np.pi,2*np.pi], ['0','$\mathdefault{\\frac{\pi}{2} }$','$\mathdefault{\pi}$','$\mathdefault{\\frac{3\pi}{2} }$','$\mathdefault{2\pi}$'])
    plt.xticks([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi],
               ['0', '$\mathdefault{0.5\pi}$', '$\mathdefault{\pi}$', '$\mathdefault{1.5\pi}$',
                '$\mathdefault{2\pi}$'])
    plt.yticks([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi],
               ['0', '$\mathdefault{0.5\pi}$', '$\mathdefault{\pi}$', '$\mathdefault{1.5\pi}$',
                '$\mathdefault{2\pi}$'])
    # plt.yticks([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi],
    #            ['0', '$\mathdefault{\\frac{\pi}{2} }$', '$\mathdefault{\pi}$', '$\mathdefault{\\frac{3\pi}{2} }$',
    #             '$\mathdefault{2\pi}$'])

    # plt.savefig(file_path + 'angle_pair_'+file_name + '.png')
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.grid(True)

    # 坐标轴粗细
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # 坐标轴刻度字体大小
    plt.xticks(fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')

    plt.xticks([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi],
               ['0', '$\mathdefault{0.5\pi}$', '$\mathdefault{\pi}$', '$\mathdefault{1.5\pi}$',
                '$\mathdefault{2\pi}$'])
    plt.yticks([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi],
               ['0', '$\mathdefault{0.5\pi}$', '$\mathdefault{\pi}$', '$\mathdefault{1.5\pi}$',
                '$\mathdefault{2\pi}$'])

    # 设置坐标轴标签
    ax.set_xlabel('Current_vector_angle (rad)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Rotor_position_angle (rad)', fontsize=15, fontweight='bold')

    ax.scatter(angle_pair[:, 0][::50], angle_pair[:, 1][::50], s=12, color='black')

    plt.show()

    # 将左上角的几个点修正
    for i in range(len(angle_pair)):
        if (angle_pair[i][0] < 1) & (angle_pair[i][1] > 6.28):
            angle_pair[i][1] = 0

    return angle_pair

# angle_pair = create_dataset_angle_pair_line(file_path,"20230708-X0+-X100+-F500-0001.csv",1)
# angle_pair = create_dataset_angle_pair_line(file_path,"20230616-Y0+-Y100-F500-0001.csv",1)

def fit_angle_pair_correlation(angle_pair):
    # 使用polyfit函数拟合十次多项式模型
    coefficients = np.polyfit(angle_pair[:,0], angle_pair[:,1], 10)

    return coefficients

def train_angle_pair_model(file_path,file_name):
    angle_pair = create_dataset_angle_pair_line(file_path,file_name,1)

    coefficients = fit_angle_pair_correlation(angle_pair)

    return coefficients

def save_angle_pair_model(coefficients,file_path,file_name):
    a, b, c, d, e, f, g, h, k, m, o = coefficients
    file = open(file_path+'angle_pair_model_'+file_name+'.txt','w')
    file.write(str(a)+','+str(b)+','+str(c)+','+str(d)+','+str(e)+','+str(f)+','+str(g)+','+str(h)+','+str(k)+','+str(m)+','+str(o)+'\n')
    file.close()

def load_angle_pair_model(file_path,file_name):
    file = open(file_path + 'angle_pair_model_' + file_name + '.txt')
    data = file.readlines()
    #print(data)
    coefficients = data[0].strip().split(',')
    file.close()
    return np.array(coefficients,dtype=float)


def fit_current_angle(current_angle, coefficients):
    '''
    current_angle shape:(2,time_n)
    '''
    a, b, c, d, e, f, g, h, k, m, o = coefficients
    print(a, b, c, d, e, f, g, h, k, m, o)

    current_angle[1] = a * current_angle[1] ** 10 + b * current_angle[1] ** 9 + c * current_angle[
        1] ** 8 + d * current_angle[1] ** 7 + e * current_angle[1] ** 6 + f * current_angle[1] ** 5 + g * \
                          current_angle[1] ** 4 + h * current_angle[1] ** 3 + k * current_angle[1] ** 2 + m * \
                          current_angle[1] ** 1 + o

    return current_angle

def plot_angle_pair_model(angle_pair, coefficients):
    a, b, c, d, e, f, g, h, k, m, o = coefficients
    print(a, b, c, d, e, f, g, h, k, m, o)
    # 生成拟合曲线的x轴数据
    x_fit = np.linspace(0, 2 * np.pi, 10000)
    # 计算拟合曲线的y轴数据
    y_fit = a * x_fit ** 10 + b * x_fit ** 9 + c * x_fit ** 8 + d * x_fit ** 7 + e * x_fit ** 6 + f * x_fit ** 5 + g * x_fit ** 4 + h * x_fit ** 3 + k * x_fit ** 2 + m * x_fit ** 1 + o
    # 绘制原始数据和拟合曲线
    plt.figure(figsize=(20,5))
    plt.scatter(angle_pair[:,0], angle_pair[:,1], label='Original Data')
    plt.plot(x_fit, y_fit, 'r', label='Fitted Curve')
    plt.legend()
    plt.show()

def split_end_part(file_name, diff_1, current_angle_sampled, joint):
    '''
    将100次重复实验的电流文件，分隔开成100份，每次重复实验中间隔4秒钟
    :param file_name:
    :param diff_1:
    :param current_angle_sampled:
    :param joint:
    :return:
    '''
    x_static = []
    for i in range(diff_1.shape[2]):
        if diff_1[joint][1][i] < 0.17:
            x_static.append(i)
    # print(len(x_static))

    tmp = [[x_static[0]]]
    for i in range(1, len(x_static)):
        if x_static[i] - tmp[-1][-1] > 10:
            tmp.append([x_static[i]])
        else:
            tmp[-1].append(x_static[i])
    print(len(tmp))

    x_static_u = []
    for i in tmp:
        # print(len(i))
        if len(i) > 700:
            x_static_u.append(i)
    print(len(x_static_u))

    index = []
    for i in range(len(x_static_u)):
        # index.append([x_static_u[i][0],x_static_u[i][-1]])
        index.append([diff_1[joint][0][x_static_u[i][0]], diff_1[joint][0][x_static_u[i][-1]]])
        # index.append([diff_1[0][0][x_static_u[i][0]],diff_1[0][0][x_static_u[i][-1]]])
        # print(diff_1[0][0][x_static_u[i+1][0]] - diff_1[0][0][x_static_u[i][-1]])
        # print(diff_1[0][0][x_static_u[i][-1]]-diff_1[0][0][x_static_u[i][0]])
        # print(diff_1[0][0][x_static_u[i][0]],diff_1[0][0][x_static_u[i][-1]],diff_1[0][0][x_static_u[i][-1]]-diff_1[0][0][x_static_u[i][0]])
        # print(diff_1[0][0][x_static_u[i][-1]],diff_1[0][0][x_static_u[i+1][0]],diff_1[0][0][x_static_u[i+1][0]] - diff_1[0][0][x_static_u[i][-1]])
    index = index[1:]
    print(len(index))

    folder_path = train_file_path + file_name + '_' + str(joint)
    if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folder_path)
    print(folder_path)

    h5f = h5py.File(train_file_path + file_name + '_' + str(joint) + '.h5', 'w')
    for i in range(int(len(index) / 2)):
        print(index[2 * i][-1])
        print(plot_current_data.get_index_at_t_v2(diff_1[joint], index[2 * i][-1]))
        current_angle_sampled_tmp = current_angle_sampled[:, :, plot_current_data.get_index_at_t_v2(current_angle_sampled[joint],
                                                                                  index[2 * i][
                                                                                      -1]) + 100:plot_current_data.get_index_at_t_v2(
            current_angle_sampled[joint], index[2 * i + 1][-1]) - 100]
        # current_angle_sampled_tmp = current_angle_sampled[:,:,index[2*i][-1]+100:index[2*i+1][-1]-100]
        # h5f.create_dataset('current_angle_sampled', data=current_angle_sampled_tmp)
        h5f.create_dataset('current_angle_' + str(round(0.025 * i, 3)), data=current_angle_sampled_tmp)
        start_proportion = 0
        end_proportion = 1
        plt = plot_current_data.create_figure((40, 5), 'time', 'current_angle')
        plt = plot_current_data.plot_series_with_time(plt, current_angle_sampled_tmp[joint][:,
                                                           int(start_proportion * current_angle_sampled_tmp.shape[
                                                               2]):int(
                                                               end_proportion * current_angle_sampled_tmp.shape[2])],
                                                      "current_angle_sampled_tmp")
        plt.savefig(folder_path + '/' + 'current_angle_' + str(round(0.025 * i, 3)) + '.png')
    h5f.close()

def split_end_part_v2(file_name, joint):
    '''
    将100次重复实验的电流文件，分隔开成100份，每次重复实验中间隔4秒钟,用来分割反向运动的电流
    :param file_name:
    :param diff_1:
    :param current_angle_sampled:
    :param joint:
    :return:
    '''

    gain_factor = 100
    sliding_window = 500
    f = h5py.File(train_file_path + 'position_from_current_angle/' + file_name + '_position_from_current_angle' + ".h5","r")
    position_from_current_angle = np.array(f['position_from_current_angle'])
    f.close()

    f = h5py.File(train_file_path + 'current_angle/' + file_name + '_current_angle' + ".h5","r")
    current_angle = np.array(f['current_angle'])
    f.close()

    position_from_current_angle_sampled_smoothed = np.array(
        [[position_from_current_angle[0][0],
          plot_current_data.moving_average(position_from_current_angle[0][1], sliding_window)],
         [position_from_current_angle[1][0],
          plot_current_data.moving_average(position_from_current_angle[1][1], sliding_window)]])

    position_from_current_angle_sampled_smoothed = position_from_current_angle_sampled_smoothed[:, :,
                                                   sliding_window:-1 * sliding_window]

    # 求轨迹的差分（相当于每一时刻的速度），为了判断静止还是运动
    diff_x = np.diff(position_from_current_angle_sampled_smoothed[0][1]) * gain_factor
    diff_y = np.diff(position_from_current_angle_sampled_smoothed[1][1]) * gain_factor

    diff_x = np.abs(diff_x)
    diff_y = np.abs(diff_y)

    diff_1 = np.array([[position_from_current_angle_sampled_smoothed[0][0][:-1], diff_x],
                       [position_from_current_angle_sampled_smoothed[1][0][:-1], diff_y]])
    print("diff_1.shape: ", diff_1.shape)

    x_static = []
    for i in range(diff_1.shape[2]):
        if diff_1[joint][1][i] < 0.005:  # 0.17
            x_static.append(diff_1[joint][0][i])
    # print(len(x_static))

    tmp = [[x_static[0]]]
    for i in range(1, len(x_static)):
        if x_static[i] - tmp[-1][-1] > 50:#20
            tmp.append([x_static[i]])
        else:
            tmp[-1].append(x_static[i])
    print(len(tmp))

    x_static_u = []
    for i in tmp:
        # print(len(i))
        if (i[-1] - i[0]) > 1200:
            x_static_u.append(i)
    print(len(x_static_u))

    x_static_u = x_static_u[1:]

    folder_path = train_file_path + file_name + '_' + str(joint)
    if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folder_path)
    print(folder_path)

    h5f = h5py.File(train_file_path + file_name + '_' + str(joint) + '.h5', 'w')
    for i in range(len(x_static_u)):
        current_angle_sampled_tmp = current_angle[:, :,
                                    plot_current_data.get_index_at_t_v2(
                                        current_angle[joint],
                                        x_static_u[i][0]) - 40000:plot_current_data.get_index_at_t_v2(
                                        current_angle[joint], x_static_u[i][-1] -500)]

        h5f.create_dataset('current_angle_' + str(round(0.025 * i, 3)), data=current_angle_sampled_tmp)
        start_proportion = 0
        end_proportion = 1
        plt = plot_current_data.create_figure((40, 5), 'time', 'current_angle')
        plt = plot_current_data.plot_series_with_time(plt, current_angle_sampled_tmp[joint][:,
                                                           int(start_proportion * current_angle_sampled_tmp.shape[
                                                               2]):int(
                                                               end_proportion * current_angle_sampled_tmp.shape[2])],
                                                      "current_angle")
        plt.savefig(folder_path + '/' + 'current_angle_' + str(round(0.025 * i, 3)) + '.png')
    h5f.close()

def split_end_part_v3(file_name, joint):
    '''
    将100次重复实验的电流文件，分隔开成100份，每次重复实验中间隔4秒钟,用来分割负向启动正向结束（正向启动负向结束）的电流
    :param file_name:
    :param diff_1:
    :param current_angle_sampled:
    :param joint:
    :return:
    '''

    gain_factor = 100
    sliding_window = 500
    f = h5py.File(train_file_path + 'position_from_current_angle/' + file_name + '_position_from_current_angle' + ".h5","r")
    position_from_current_angle = np.array(f['position_from_current_angle'])
    f.close()

    f = h5py.File(train_file_path + 'current_angle/' + file_name + '_current_angle' + ".h5","r")
    current_angle = np.array(f['current_angle'])
    f.close()

    position_from_current_angle_sampled_smoothed = np.array(
        [[position_from_current_angle[0][0],
          plot_current_data.moving_average(position_from_current_angle[0][1], sliding_window)],
         [position_from_current_angle[1][0],
          plot_current_data.moving_average(position_from_current_angle[1][1], sliding_window)]])

    position_from_current_angle_sampled_smoothed = position_from_current_angle_sampled_smoothed[:, :,
                                                   sliding_window:-1 * sliding_window]

    # 求轨迹的差分（相当于每一时刻的速度），为了判断静止还是运动
    diff_x = np.diff(position_from_current_angle_sampled_smoothed[0][1]) * gain_factor
    diff_y = np.diff(position_from_current_angle_sampled_smoothed[1][1]) * gain_factor

    diff_x = np.abs(diff_x)
    diff_y = np.abs(diff_y)

    diff_1 = np.array([[position_from_current_angle_sampled_smoothed[0][0][:-1], diff_x],
                       [position_from_current_angle_sampled_smoothed[1][0][:-1], diff_y]])
    print("diff_1.shape: ", diff_1.shape)

    x_static = []
    for i in range(diff_1.shape[2]):
        if diff_1[joint][1][i] < 0.005:  # 0.17
            x_static.append(diff_1[joint][0][i])
    # print(len(x_static))

    tmp = [[x_static[0]]]
    for i in range(1, len(x_static)):
        if x_static[i] - tmp[-1][-1] > 50:#20
            tmp.append([x_static[i]])
        else:
            tmp[-1].append(x_static[i])
    print(len(tmp))

    x_static_u = []
    for i in tmp:
        # print(len(i))
        if (i[-1] - i[0]) > 1200:
            x_static_u.append(i)
    print(len(x_static_u))

    x_static_u = x_static_u[1:]

    folder_path = train_file_path + file_name + '_' + str(joint)
    if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folder_path)
    print(folder_path)

    h5f = h5py.File(train_file_path + file_name + '_' + str(joint) + '.h5', 'w')
    for i in range(int(len(x_static_u)/2)):
        current_angle_sampled_tmp = current_angle[:, :,
                                    plot_current_data.get_index_at_t_v2(
                                        current_angle[joint],
                                        x_static_u[2*i][0] + 1000):plot_current_data.get_index_at_t_v2(
                                        current_angle[joint], x_static_u[2*i+1][-1]) -1000]

        h5f.create_dataset('current_angle_' + str(round(0.025 * i, 3)), data=current_angle_sampled_tmp)
        start_proportion = 0
        end_proportion = 1
        plt = plot_current_data.create_figure((40, 5), 'time', 'current_angle')
        plt = plot_current_data.plot_series_with_time(plt, current_angle_sampled_tmp[joint][:,
                                                           int(start_proportion * current_angle_sampled_tmp.shape[
                                                               2]):int(
                                                               end_proportion * current_angle_sampled_tmp.shape[2])],
                                                      "current_angle")
        plt.savefig(folder_path + '/' + 'current_angle_' + str(round(0.025 * i, 3)) + '.png')
    h5f.close()

def load_end_part(file_name,joint):
    f = h5py.File(train_file_path+file_name+".h5", "r")
    for i in range(101):
        print(np.array(f['current_angle_'+str(round(0.025*i,3))]).shape)
    f.close()

def filter_abnormal_points_angle_jump(current_angle_sampled,num_joints):
    '''
    当角度转一周，从0到2pi跃变时，删除中间出现的异常点
    '''

    index = []
    for i in range(num_joints):
        for j in range(1, current_angle_sampled.shape[2] - 1):
            if abs(current_angle_sampled[i][1][j + 1] - current_angle_sampled[i][1][j - 1]) > 6.15:
                index.append(j)
    index = list(set(index))

    current_angle_sampled = np.delete(current_angle_sampled,index,axis=2)

    return current_angle_sampled

def extract_end_point(file_name, joint, sliding_window):
    f = h5py.File(train_file_path + file_name + ".h5", "r")

    re = []

    for key in f.keys():
        print(key)
        current_angle = np.array(f[key])
        print("current_angle.shape: ", current_angle.shape)

        position_from_current_angle = plot_current_data.angle_to_position_v4(current_angle, num_joints, pole_pair,
                                                                             screw_lead, [0, 0])
        position_from_current_angle = plot_current_data.position_negate(position_from_current_angle, num_joints)
        print("position_from_current_angle.shape: ", position_from_current_angle.shape)

        position_from_current_angle_sampled = plot_current_data.downsample_v2(position_from_current_angle, num_joints,
                                                                              1)

        print("position_from_current_angle_sampled.shape: ", position_from_current_angle_sampled.shape)

        position_from_current_angle_sampled_smoothed = np.array(
            [[position_from_current_angle_sampled[0][0],
              plot_current_data.moving_average(position_from_current_angle_sampled[0][1], sliding_window)],
             [position_from_current_angle_sampled[1][0],
              plot_current_data.moving_average(position_from_current_angle_sampled[1][1], sliding_window)]])

        position_from_current_angle_sampled_smoothed = position_from_current_angle_sampled_smoothed[:, :,
                                                       sliding_window:-1 * sliding_window]

        print("position_from_current_angle_sampled_smoothed.shape: ",
              position_from_current_angle_sampled_smoothed.shape)

        from scipy.signal import find_peaks

        peaks, properties = find_peaks(
            position_from_current_angle_sampled_smoothed[joint][1])  # prominence=0.01, width=5

        if float(key.split('_')[-1]) > 1.05:
            minima = plot_current_data.get_minima(
                position_from_current_angle_sampled_smoothed[joint][1][peaks[0]:peaks[1]])
            if len(minima[0]) != 0:
                peaks[0] = minima[0][-1] + peaks[0]
        else:
            # print('small')
            peaks = peaks[1:]

        #     %matplotlib notebook
        plt.figure(figsize=(40,10))
        #plt.figure()
        plt.plot(position_from_current_angle[joint][0],position_from_current_angle[joint][1])
        plt.plot(position_from_current_angle_sampled_smoothed[joint][0],position_from_current_angle_sampled_smoothed[joint][1])
        plt.plot(current_angle[joint][0],current_angle[joint][1])
        # plt.scatter(position_from_current_angle_sampled_smoothed[joint][0][1337],position_from_current_angle_sampled_smoothed[joint][1][1337])
        for i in peaks:
            plt.scatter(position_from_current_angle_sampled_smoothed[joint][0][i],position_from_current_angle_sampled_smoothed[joint][1][i])
            #plt.scatter(position_from_current_angle_sampled[joint][0][i],position_from_current_angle_sampled[joint][1][i])
        #plt.show()
        plt.savefig(train_file_path + file_name+'/'+key+'.png')

        dif = position_from_current_angle_sampled_smoothed[joint][1][peaks[1]] - np.mean(
            position_from_current_angle_sampled_smoothed[joint][1][-1000:])
        # print(dif)
        # print(np.mean(current_angle[joint][1][-20000:]))
        # print(np.mean(current_angle[joint][1][-20000:]) - (dif/2.5)*2*np.pi)
        if np.std(current_angle[joint][1][-20000:]) > 1:
            # file.write(str(2*np.pi - (dif/2.5)*2*np.pi)+','+str(2*np.pi)+'\n')
            re.append([float(key.split('_')[-1]), 2 * np.pi - (dif / 2.5) * 2 * np.pi, 2 * np.pi])
        else:
            # file.write(str(np.mean(current_angle[joint][1][-20000:]) - (dif/2.5)*2*np.pi)+','+str(np.mean(current_angle[joint][1][-20000:]))+'\n')
            re.append([float(key.split('_')[-1]), np.mean(current_angle[joint][1][-20000:]) - (dif / 2.5) * 2 * np.pi,
                       np.mean(current_angle[joint][1][-20000:])])

    f.close()

    re = sorted(re, key=lambda x: x[0])
    file = open(train_file_path + file_name + '.txt', 'w')
    for i in range(len(re)):
        file.write(str(re[i][0]) + ',' + str(re[i][1]) + ',' + str(re[i][2]) + '\n')
    file.close()

def extract_end_point_v2(file_name, init_state, joint, sliding_window, gain_factor):
    '''
    用来提取负向运动结束时的电流角
    :param file_name:
    :param joint:
    :param sliding_window:
    :param gain_factor:
    :return:
    '''
    f = h5py.File(train_file_path + file_name + ".h5", "r")

    re = []

    for key in f.keys():
        print(key)
        current_angle = np.array(f[key])
        # print("current_angle.shape: ", current_angle.shape)

        position_from_current_angle = plot_current_data.angle_to_position_v4(current_angle, num_joints, pole_pair,
                                                                             screw_lead, [0, 0])
        position_from_current_angle = plot_current_data.position_negate(position_from_current_angle, num_joints)
        # print("position_from_current_angle.shape: ", position_from_current_angle.shape)

        position_from_current_angle_sampled = plot_current_data.downsample_v2(position_from_current_angle, num_joints,
                                                                              1)

        # print("position_from_current_angle_sampled.shape: ", position_from_current_angle_sampled.shape)

        position_from_current_angle_sampled_smoothed = np.array(
            [[position_from_current_angle_sampled[0][0],
              plot_current_data.moving_average(position_from_current_angle_sampled[0][1], sliding_window)],
             [position_from_current_angle_sampled[1][0],
              plot_current_data.moving_average(position_from_current_angle_sampled[1][1], sliding_window)]])

        position_from_current_angle_sampled_smoothed = position_from_current_angle_sampled_smoothed[:, :,
                                                       sliding_window:-1 * sliding_window]

        # print("position_from_current_angle_sampled_smoothed.shape: ",position_from_current_angle_sampled_smoothed.shape)

        # 求轨迹的差分（相当于每一时刻的速度），为了判断静止还是运动
        diff_x = np.diff(position_from_current_angle_sampled_smoothed[0][1]) * gain_factor
        diff_y = np.diff(position_from_current_angle_sampled_smoothed[1][1]) * gain_factor

        # diff_x = np.abs(diff_x)
        # diff_y = np.abs(diff_y)

        diff_1 = np.array([[position_from_current_angle_sampled_smoothed[0][0][:-1], diff_x],
                           [position_from_current_angle_sampled_smoothed[1][0][:-1], diff_y]])
        # print("diff_1.shape: ", diff_1.shape)

        x_movement = plot_current_data.judge_movement_test(diff_1[joint], init_state, 100, 0.02)  # 20,0.02
        #x_movement = plot_current_data.judge_movement_test(diff_1[joint], init_state, 50, 0.007)#20,0.02
        #y_movement = plot_current_data.judge_movement_test(diff_1[1], 50, 0.007)#20,0.02

        for i in range(x_movement.shape[1] - 1):
            if ((x_movement[1][i] == 3) & (x_movement[1][i+1] == 1)) | ((x_movement[1][i] == 2) & (x_movement[1][i+1] == 0)):
                # print(x_movement[0][i+1])
                # print(plot_current_data.get_position_at_t_v2(position_from_current_angle_sampled_smoothed[joint],x_movement[0][i + 1]))
                dif = (plot_current_data.get_position_at_t_v2(position_from_current_angle_sampled_smoothed[joint],x_movement[0][i + 1])) - np.mean(position_from_current_angle_sampled_smoothed[joint][1][-1000:])
                re.append([float(key.split('_')[-1]), np.mean(current_angle[joint][1][-20000:]) - (dif / 2.5) * 2 * np.pi,np.mean(current_angle[joint][1][-20000:])])
                break
        # plt.figure(figsize=(40,10))
        # #plt.figure()
        # plt.plot(position_from_current_angle[joint][0],position_from_current_angle[joint][1])
        # plt.plot(position_from_current_angle_sampled_smoothed[joint][0],position_from_current_angle_sampled_smoothed[joint][1])
        # plt.plot(current_angle[joint][0],current_angle[joint][1])
        # # plt.scatter(position_from_current_angle_sampled_smoothed[joint][0][1337],position_from_current_angle_sampled_smoothed[joint][1][1337])
        # for i in peaks:
        #     plt.scatter(position_from_current_angle_sampled_smoothed[joint][0][i],position_from_current_angle_sampled_smoothed[joint][1][i])
        #     #plt.scatter(position_from_current_angle_sampled[joint][0][i],position_from_current_angle_sampled[joint][1][i])
        # #plt.show()
        # plt.savefig(train_file_path + file_name+'/'+key+'.png')

    f.close()

    # re = sorted(re, key=lambda x: x[0])
    # file = open(train_file_path + file_name + '.txt', 'w')
    # for i in range(len(re)):
    #     file.write(str(re[i][0]) + ',' + str(re[i][1]) + ',' + str(re[i][2]) + '\n')
    # file.close()

def extract_end_point_v3(file_name, joint):
    '''
    用来提取负向提取，正向结束时的电流角，启动静止时的电流角、制动静止时的电流角
    :param file_name:
    :param joint:
    :return:
    '''
    f = h5py.File(train_file_path + file_name + ".h5", "r")

    re = []

    for key in f.keys():
        print(key)
        current_angle = np.array(f[key])
        # print("current_angle.shape: ", current_angle.shape)
        re.append([float(key.split('_')[-1]), np.mean(current_angle[joint][1][:20000]), np.mean(current_angle[joint][1][-20000:])])
    f.close()

    re = sorted(re, key=lambda x: x[0])
    file = open(train_file_path + file_name + '.txt', 'w')
    for i in range(len(re)):
        file.write(str(re[i][0]) + ',' + str(re[i][1]) + ',' + str(re[i][2]) + '\n')
    file.close()

def extract_end_point_v4(file_name, init_state, joint, sliding_window, gain_factor):
    '''
    用来提取负向提取，正向结束时的电流角，启动静止时的电流角、制动静止时的电流角
    :param file_name:
    :param joint:
    :return:
    '''
    f = h5py.File(train_file_path + file_name + ".h5", "r")

    re = []

    for key in f.keys():
        print(key)
        current_angle = np.array(f[key])
        # print("current_angle.shape: ", current_angle.shape)

        position_from_current_angle = plot_current_data.angle_to_position_v4(current_angle, num_joints, pole_pair,
                                                                             screw_lead, [0, 0])
        position_from_current_angle = plot_current_data.position_negate(position_from_current_angle, num_joints)
        # print("position_from_current_angle.shape: ", position_from_current_angle.shape)

        position_from_current_angle_sampled = plot_current_data.downsample_v2(position_from_current_angle, num_joints,
                                                                              1)

        # print("position_from_current_angle_sampled.shape: ", position_from_current_angle_sampled.shape)

        position_from_current_angle_sampled_smoothed = np.array(
            [[position_from_current_angle_sampled[0][0],
              plot_current_data.moving_average(position_from_current_angle_sampled[0][1], sliding_window)],
             [position_from_current_angle_sampled[1][0],
              plot_current_data.moving_average(position_from_current_angle_sampled[1][1], sliding_window)]])

        position_from_current_angle_sampled_smoothed = position_from_current_angle_sampled_smoothed[:, :,
                                                       sliding_window:-1 * sliding_window]

        # print("position_from_current_angle_sampled_smoothed.shape: ",position_from_current_angle_sampled_smoothed.shape)

        # 求轨迹的差分（相当于每一时刻的速度），为了判断静止还是运动
        diff_x = np.diff(position_from_current_angle_sampled_smoothed[0][1]) * gain_factor
        diff_y = np.diff(position_from_current_angle_sampled_smoothed[1][1]) * gain_factor


        diff_1 = np.array([[position_from_current_angle_sampled_smoothed[0][0][:-1], diff_x],
                           [position_from_current_angle_sampled_smoothed[1][0][:-1], diff_y]])
        # print("diff_1.shape: ", diff_1.shape)

        x_movement = plot_current_data.judge_movement_test(diff_1[joint], init_state, 100, 0.02)  # 20,0.02

        for i in range(x_movement.shape[1] - 1):
            if ((x_movement[1][i][0] == 3) & (x_movement[1][i+1][0] == 1)) | ((x_movement[1][i][0] == 2) & (x_movement[1][i+1][0] == 0)):
                dif = (plot_current_data.get_position_at_t_v2(position_from_current_angle_sampled_smoothed[joint],x_movement[0][i + 1])) - np.mean(position_from_current_angle_sampled_smoothed[joint][1][-1000:])
                re.append([float(key.split('_')[-1]), np.mean(current_angle[joint][1][:20000]), np.mean(current_angle[joint][1][-20000:]) - (dif / 2.5) * 2 * np.pi])
                #print(re[-1])
                break

    f.close()

    re = sorted(re, key=lambda x: x[0])
    file = open(train_file_path + file_name + '.txt', 'w')
    for i in range(len(re)):
        file.write(str(re[i][0]) + ',' + str(re[i][1]) + ',' + str(re[i][2]) + '\n')
    file.close()

def plot_end_angle(file_name,key,joint,sliding_window):
    f = h5py.File(train_file_path + file_name + ".h5", "r")
    print(key)
    current_angle = np.array(f[key])
    print("current_angle.shape: ", current_angle.shape)

    position_from_current_angle = plot_current_data.angle_to_position_v4(current_angle, num_joints, pole_pair,
                                                                         screw_lead, [0, 0])
    position_from_current_angle = plot_current_data.position_negate(position_from_current_angle, num_joints)
    print("position_from_current_angle.shape: ", position_from_current_angle.shape)

    position_from_current_angle_sampled = plot_current_data.downsample_v2(position_from_current_angle, num_joints, 1)

    print("position_from_current_angle_sampled.shape: ", position_from_current_angle_sampled.shape)

    position_from_current_angle_sampled_smoothed = np.array(
        [[position_from_current_angle_sampled[0][0],
          plot_current_data.moving_average(position_from_current_angle_sampled[0][1], sliding_window)],
         [position_from_current_angle_sampled[1][0],
          plot_current_data.moving_average(position_from_current_angle_sampled[1][1], sliding_window)]])

    position_from_current_angle_sampled_smoothed = position_from_current_angle_sampled_smoothed[:, :,
                                                   sliding_window:-1 * sliding_window]

    print("position_from_current_angle_sampled_smoothed.shape: ", position_from_current_angle_sampled_smoothed.shape)

    # plt.figure(figsize=(40,10))
    plt.figure()
    plt.plot(position_from_current_angle[joint][0], position_from_current_angle[joint][1])
    plt.plot(position_from_current_angle_sampled_smoothed[joint][0],
             position_from_current_angle_sampled_smoothed[joint][1])
    plt.plot(current_angle[joint][0], current_angle[joint][1])
    plt.show()


    f.close()
    return


# split_end_part(file_name,diff_1,current_angle,0)
# split_end_part(file_name,diff_1,current_angle,1)
#
# file_name = '20230720-cycle-0.025-10-F500-xy-0001_0'
# extract_end_point(file_name,0,25)
#
# file_name = '20230720-cycle-0.025-10-F500-xy-0001_1'
# extract_end_point(file_name,1,25)



# file_name = '20230711-cycle-0.025-10-F500-0001'
# extract_end_point(file_name,1,25)
#
# file_name = '20230711-cycle-0.025-10-F2000-x-0001'
# extract_end_point(file_name,0,25)
#
# file_name = '20230711-cycle-0.025-10-F2000-0001'
# extract_end_point(file_name,1,25)

# file_name = '20230711-cycle-0.025-10-F2000-arc-0001'
# extract_end_point(file_name,1,25)
#
# file_name = '20230711-cycle-0.025-10-F500-arc-0001'
# extract_end_point(file_name,1,25)

# file_name = '20230711-cycle-0.025-10-F1000-0001'
# extract_end_point(file_name,1,25)
#
# file_name = '20230711-cycle-0.025-10-F1500-0001'
# extract_end_point(file_name,1,25)
#
# file_name = '20230711-cycle-0.025-10-F1000-x-0001'
# extract_end_point(file_name,0,25)
#
# file_name = '20230711-cycle-0.025-10-F1500-x-0001'
# extract_end_point(file_name,0,25)

# file_name = '20230726-cycle-0.025-10-F500-xy-negative-0001_0'
# extract_end_point(file_name,0,25)

f = open(train_file_path+'20230726-cycle-0.025-10-F500-xy-negative-0001_1'+'.txt','r')
data_F500xy_negative_y = f.readlines()
re_F500xy_negative_y = []
for i in range(len(data_F500xy_negative_y)):
    re_F500xy_negative_y.append(data_F500xy_negative_y[i].strip().split(','))
f.close()
re_F500xy_negative_y = np.array(re_F500xy_negative_y,dtype = float)
print(re_F500xy_negative_y.shape)

f = open(train_file_path+'20230726-cycle-0.025-10-F500-xy-negative-0001_0'+'.txt','r')
data_F500xy_negative_x = f.readlines()
re_F500xy_negative_x = []
for i in range(len(data_F500xy_negative_x)):
    re_F500xy_negative_x.append(data_F500xy_negative_x[i].strip().split(','))
f.close()
re_F500xy_negative_x = np.array(re_F500xy_negative_x,dtype = float)
print(re_F500xy_negative_x.shape)

re_F500xy_negative = [re_F500xy_negative_x,re_F500xy_negative_y]

# plt.figure()
# plt.title('re_F500xy_negative_x')
# plt.plot(re_F500xy_negative_x[:,0],re_F500xy_negative_x[:,1],label = 'end_1_re_F500xy_negative_x')
# plt.plot(re_F500xy_negative_x[:,0],re_F500xy_negative_x[:,2],label = 'end_2_re_F500xy_negative_x')
# plt.legend()
#
# plt.figure()
# plt.title('re_F500xy_negative_y')
# plt.plot(re_F500xy_negative_y[:,0],re_F500xy_negative_y[:,1],label = 'end_1_re_F500xy_negative_y')
# plt.plot(re_F500xy_negative_y[:,0],re_F500xy_negative_y[:,2],label = 'end_2_re_F500xy_negative_y')
# plt.legend()
# plt.show()

# f = open(train_file_path+'20230711-cycle-0.025-10-F500-0001'+'.txt','r')
# data_F500_y = f.readlines()
# re_F500_y = []
# for i in range(len(data_F500_y)):
#     re_F500_y.append(data_F500_y[i].strip().split(','))
# f.close()
# re_F500_y = np.array(re_F500_y,dtype = float)
# print(re_F500_y.shape)
# re_F500_y = sorted(re_F500_y, key=lambda x: x[2])
# re_F500_y = np.array(re_F500_y,dtype = float)
#
# f = open(train_file_path+'20230711-cycle-0.025-10-F1000-0001'+'.txt','r')
# data_F1000_y = f.readlines()
# re_F1000_y = []
# for i in range(len(data_F1000_y)):
#     re_F1000_y.append(data_F1000_y[i].strip().split(','))
# f.close()
# re_F1000_y = np.array(re_F1000_y,dtype = float)
# print(re_F1000_y.shape)
# re_F1000_y = sorted(re_F1000_y, key=lambda x: x[2])
# re_F1000_y = np.array(re_F1000_y,dtype = float)
#
# f = open(train_file_path+'20230711-cycle-0.025-10-F1500-0001'+'.txt','r')
# data_F1500_y = f.readlines()
# re_F1500_y = []
# for i in range(len(data_F1500_y)):
#     re_F1500_y.append(data_F1500_y[i].strip().split(','))
# f.close()
# re_F1500_y = np.array(re_F1500_y,dtype = float)
# print(re_F1500_y.shape)
# re_F1500_y = sorted(re_F1500_y, key=lambda x: x[2])
# re_F1500_y = np.array(re_F1500_y,dtype = float)
#
# f = open(train_file_path+'20230711-cycle-0.025-10-F2000-0001'+'.txt','r')
# data_F2000_y = f.readlines()
# re_F2000_y = []
# for i in range(len(data_F2000_y)):
#     re_F2000_y.append(data_F2000_y[i].strip().split(','))
# f.close()
# re_F2000_y = np.array(re_F2000_y,dtype = float)
# print(re_F2000_y.shape)
#
# re_F2000_y = sorted(re_F2000_y, key=lambda x: x[2])
# re_F2000_y = np.array(re_F2000_y,dtype = float)

f = open(train_file_path+'20230720-cycle-0.025-10-F500-xy-0001_1'+'.txt','r')
data_F500xy_y = f.readlines()
re_F500xy_y = []
for i in range(len(data_F500xy_y)):
    re_F500xy_y.append(data_F500xy_y[i].strip().split(','))
f.close()
re_F500xy_y = np.array(re_F500xy_y,dtype = float)
print(re_F500xy_y.shape)

f = open(train_file_path+'20230720-cycle-0.025-10-F500-xy-0001_0'+'.txt','r')
data_F500xy_x = f.readlines()
re_F500xy_x = []
for i in range(len(data_F500xy_x)):
    re_F500xy_x.append(data_F500xy_x[i].strip().split(','))
f.close()
re_F500xy_x = np.array(re_F500xy_x,dtype = float)
print(re_F500xy_x.shape)

re_F500xy = [re_F500xy_x,re_F500xy_y]




f = open(train_file_path+'20230731-start+-end--F500-50hz-0001_1'+'.txt','r')
data_F500xy_y = f.readlines()
f2r_F500xy_y = []
for i in range(len(data_F500xy_y)):
    f2r_F500xy_y.append(data_F500xy_y[i].strip().split(','))
f.close()
f2r_F500xy_y = np.array(f2r_F500xy_y,dtype = float)
print(f2r_F500xy_y.shape)

f = open(train_file_path+'20230731-start+-end--F500-50hz-0001_0'+'.txt','r')
data_F500xy_x = f.readlines()
f2r_F500xy_x = []
for i in range(len(data_F500xy_x)):
    f2r_F500xy_x.append(data_F500xy_x[i].strip().split(','))
f.close()
f2r_F500xy_x = np.array(f2r_F500xy_x,dtype = float)
print(f2r_F500xy_x.shape)

f2r_F500xy = [f2r_F500xy_x,f2r_F500xy_y]



f = open(train_file_path+'20230731-start--end+-F500-50hz-0001_1'+'.txt','r')
data_F500xy_y = f.readlines()
r2f_F500xy_y = []
for i in range(len(data_F500xy_y)):
    r2f_F500xy_y.append(data_F500xy_y[i].strip().split(','))
f.close()
r2f_F500xy_y = np.array(r2f_F500xy_y,dtype = float)
print(r2f_F500xy_y.shape)

f = open(train_file_path+'20230731-start--end+-F500-50hz-0001_0'+'.txt','r')
data_F500xy_x = f.readlines()
r2f_F500xy_x = []
for i in range(len(data_F500xy_x)):
    r2f_F500xy_x.append(data_F500xy_x[i].strip().split(','))
f.close()
r2f_F500xy_x = np.array(r2f_F500xy_x,dtype = float)
print(r2f_F500xy_x.shape)

r2f_F500xy = [r2f_F500xy_x,r2f_F500xy_y]


# re_F500xy_y = sorted(re_F500xy_y, key=lambda x: x[2])
# re_F500xy_y = np.array(re_F500xy_y,dtype = float)

# plt.figure()
# #plt.plot(re_F500_y[:,0],re_F500_y[:,1],label = 'end_1_F500_y')
# #plt.plot(re_F500_y[:,0],re_F500_y[:,2],label = 'end_2_F500_y')
#
# #plt.plot(re_F2000_y[:,0],re_F2000_y[:,1],label = 'end_1_F2000_y')
# #plt.plot(re_F2000_y[:,0],re_F2000_y[:,2],label = 'end_2_F2000_y')
# # plt.plot(re_F500_y[:,2],re_F2000_y[:,1],label = 'end_1_F500_y')
# # plt.plot(re_F1000_y[:,2],re_F2000_y[:,1],label = 'end_1_F1000_y')
# # plt.plot(re_F1500_y[:,2],re_F2000_y[:,1],label = 'end_1_F1500_y')
# # plt.plot(re_F2000_y[:,2],re_F2000_y[:,1],label = 'end_1_F2000_y')
# plt.plot(re_F500xy_y[:,0],re_F500xy_y[:,1],label = 'end_1_re_F500xy_y')
# plt.plot(re_F500xy_y[:,0],re_F500xy_y[:,2],label = 'end_2_re_F500xy_y')
# plt.legend()
# plt.show()

# f = open(train_file_path+'20230711-cycle-0.025-10-F500-x-0001'+'.txt','r')
# data_F500_x = f.readlines()
# re_F500_x = []
# for i in range(len(data_F500_x)):
#     re_F500_x.append(data_F500_x[i].strip().split(','))
# f.close()
# re_F500_x = np.array(re_F500_x,dtype = float)
# print(re_F500_x.shape)
# re_F500_x = sorted(re_F500_x, key=lambda x: x[2])
# re_F500_x = np.array(re_F500_x,dtype = float)
#
# f = open(train_file_path+'20230711-cycle-0.025-10-F1000-x-0001'+'.txt','r')
# data_F1000_x = f.readlines()
# re_F1000_x = []
# for i in range(len(data_F1000_x)):
#     re_F1000_x.append(data_F1000_x[i].strip().split(','))
# f.close()
# re_F1000_x = np.array(re_F1000_x,dtype = float)
# print(re_F1000_x.shape)
# re_F1000_x = sorted(re_F1000_x, key=lambda x: x[2])
# re_F1000_x = np.array(re_F1000_x,dtype = float)
#
# f = open(train_file_path+'20230711-cycle-0.025-10-F1500-x-0001'+'.txt','r')
# data_F1500_x = f.readlines()
# re_F1500_x = []
# for i in range(len(data_F1500_x)):
#     re_F1500_x.append(data_F1500_x[i].strip().split(','))
# f.close()
# re_F1500_x = np.array(re_F1500_x,dtype = float)
# print(re_F1500_x.shape)
# re_F1500_x = sorted(re_F1500_x, key=lambda x: x[2])
# re_F1500_x = np.array(re_F1500_x,dtype = float)
#
# f = open(train_file_path+'20230711-cycle-0.025-10-F2000-x-0001'+'.txt','r')
# data_F2000_x = f.readlines()
# re_F2000_x = []
# for i in range(len(data_F2000_x)):
#     re_F2000_x.append(data_F2000_x[i].strip().split(','))
# f.close()
# re_F2000_x = np.array(re_F2000_x,dtype = float)
# print(re_F2000_x.shape)
# re_F2000_x = sorted(re_F2000_x, key=lambda x: x[2])
# re_F2000_x = np.array(re_F2000_x,dtype = float)
#
# # plt.figure()
# # # plt.plot(re_F500_x[:,0],re_F500_x[:,1],label = 'end_1_F500_y')
# # # plt.plot(re_F500_x[:,0],re_F500_x[:,2],label = 'end_2_F500_y')
# # #
# # # plt.plot(re_F2000_x[:,0],re_F2000_x[:,1],label = 'end_1_F2000_y')
# # # plt.plot(re_F2000_x[:,0],re_F2000_x[:,2],label = 'end_2_F2000_y')
# # plt.plot(re_F500_x[:,2],re_F2000_x[:,1],label = 'end_1_F500_x')
# # plt.plot(re_F1000_x[:,2],re_F2000_x[:,1],label = 'end_1_F1000_x')
# # plt.plot(re_F1500_x[:,2],re_F2000_x[:,1],label = 'end_1_F1500_x')
# # plt.plot(re_F2000_x[:,2],re_F2000_x[:,1],label = 'end_1_F2000_x')
# # plt.legend()
# # plt.show()
#
# tar = 1.5
#
# for i in range(re_F500_x.shape[0]):
#     if re_F500_x[i][0] == tar:
#         print(re_F500_x[i][0])
#         print(re_F500_x[i][1])
#         print(re_F500_x[i][2])
#         print(re_F500_x[i][2] - re_F500_x[i][1])
#
# for i in range(re_F1000_x.shape[0]):
#     if re_F1000_x[i][0] == tar:
#         # print(re_F1000_x[i][1])
#         # print(re_F1000_x[i][2])
#         print(re_F1000_x[i][2] - re_F1000_x[i][1])
#
# for i in range(re_F1500_x.shape[0]):
#     if re_F1500_x[i][0] == tar:
#         # print(re_F1500_x[i][1])
#         # print(re_F1500_x[i][2])
#         print(re_F1500_x[i][2] - re_F1500_x[i][1])
#
# for i in range(re_F2000_x.shape[0]):
#     if re_F2000_x[i][0] == tar:
#         # print(re_F2000_x[i][1])
#         # print(re_F2000_x[i][2])
#         print(re_F2000_x[i][2] - re_F2000_x[i][1])
# #
# #
# #
# for i in range(re_F500_y.shape[0]):
#     if re_F500_y[i][0] == tar:
#         # print(re_F500_y[i][1])
#         # print(re_F500_y[i][2])
#         print(re_F500_y[i][2] - re_F500_y[i][1])
#
# for i in range(re_F1000_y.shape[0]):
#     if re_F1000_y[i][0] == tar:
#         # print(re_F1000_y[i][1])
#         # print(re_F1000_y[i][2])
#         print(re_F1000_y[i][2] - re_F1000_y[i][1])
#
# for i in range(re_F1500_y.shape[0]):
#     if re_F1500_y[i][0] == tar:
#         # print(re_F1500_y[i][1])
#         # print(re_F1500_y[i][2])
#         print(re_F1500_y[i][2] - re_F1500_y[i][1])
#
# for i in range(re_F2000_y.shape[0]):
#     if re_F2000_y[i][0] == tar:
#         # print(re_F2000_y[i][1])
#         # print(re_F2000_y[i][2])
#         print(re_F2000_y[i][2] - re_F2000_y[i][1])


# print(plot_current_data.find_end_offset(re_F500xy_x,0.5))
# print(plot_current_data.find_end_offset(re_F500xy_y,0.5))
#
# plot_end_angle("20230720-cycle-0.025-10-F500-xy-0001_0",'current_angle_1.5',0,25)
# plot_end_angle("20230720-cycle-0.025-10-F500-xy-0001_1",'current_angle_0.5',1,25)

# file_name = '20230731-start--end+-F500-50hz-0001'
# split_end_part_v3(file_name,0)
# split_end_part_v3(file_name,1)
# extract_end_point_v2('20230731-start--end+-F500-50hz-0001_1', [1,0], 0, 20, 1000)

# extract_end_point_v4('20230731-start--end+-F500-50hz-0001_0',[1,0], 0, 20, 1000)
#
# extract_end_point_v4('20230731-start--end+-F500-50hz-0001_1',[1,0], 1, 20, 1000)
#
# extract_end_point_v4('20230731-start+-end--F500-50hz-0001_0',[0,0], 0, 20, 1000)
#
# extract_end_point_v4('20230731-start+-end--F500-50hz-0001_1',[0,0], 1, 20, 1000)

# extract_end_point_v2('20230726-cycle-0.025-10-F500-xy-negative-0001_0', 3, 0, 20, 1000)

# extract_end_point_v2('20230726-cycle-0.025-10-F500-xy-negative-0001_1', 3, 1, 20, 1000)

# extract_end_point_v2('20230726-cycle-0.025-10-F500-xy-negative-0001_0', 1, 20, 1000)
# #
# extract_end_point_v2('20230726-cycle-0.025-10-F500-xy-negative-0001_0', 1, 20, 1000)

# file_name = '20230804-start--end+-F500-50hz-0001'
# # # split_end_part_v2(file_name,0)
# #
# split_end_part_v3(file_name, 0)
#
# split_end_part_v3(file_name, 1)
#
# file_name = '20230804-start+-end--F500-50hz-0001'
# # # split_end_part_v2(file_name,0)
# #
# split_end_part_v3(file_name, 0)
#
# split_end_part_v3(file_name, 1)

# extract_end_point_v2('20230720-cycle-0.025-10-F250-xy-0001_0', 1, 20, 1000)



