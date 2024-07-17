import numpy as np
import plot_current_data
import matplotlib.pyplot as plt
import os
import train
import h5py

screw_lead = 10
num_joints = 2
sample_time = 0.001
pole_pair = 4
sliding_window = 50

# train_file_path = 'C:/Users/lsj\PycharmProjects/experiment0616/train_data/'
# file_path = 'C:\\Users\lsj\PycharmProjects\experiment0616\\train_data\csv\\'
#
# files = ['20230804-start+-end--F500-50hz-0001','20230804-start--end+-F500-50hz-0001']


model_file_path =  'C:/Users/lsj\PycharmProjects/experiment0616/train_data/'
# train_file_path = 'C:/Users/lsj\PycharmProjects/experiment0616/jingdu_test/'
# file_path = 'C:\\Users\lsj\PycharmProjects\experiment0616\\jingdu_test\csv\\'

train_file_path = 'C:/Users/lsj\PycharmProjects/experiment0616/line/'
file_path = 'C:\\Users\lsj\PycharmProjects\experiment0616\\line\csv\\'

# train_file_path = 'C:/Users/lsj\PycharmProjects/experiment0616/line/'
# file_path = 'C:\\Users\lsj\PycharmProjects\experiment0616\\line\csv\\'


def split_end_part(file_name):
    '''
    将50次重复实验的电流文件，分隔开成50份，每次重复实验中间隔4秒钟,用来分割50次精度测试的电流
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

    # f = h5py.File(train_file_path + 'current_angle/' + file_name + '_current_angle' + ".h5","r")
    # current_angle = np.array(f['current_angle'])
    # f.close()

    f = h5py.File(train_file_path + 'current_angle/' + file_name + '_current_angle_original' + ".h5", "r")
    current_angle_original = np.array(f['current_angle_original'])
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
        if diff_1[0][1][i] < 0.005:  # 0.17
            x_static.append(diff_1[0][0][i])
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
    print(len(x_static_u))

    folder_path = train_file_path + file_name
    if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folder_path)
    print(folder_path)

    f = h5py.File(train_file_path + 'position_from_current_angle/' + file_name + '_position_from_current_angle_sample' + ".h5",
                  "r")
    position_from_current_angle_sample = np.array(f['position_from_current_angle_sample'])
    f.close()

    h5f = h5py.File(train_file_path + file_name + '.h5', 'w')
    for i in range(int(len(x_static_u)/2)):
        current_angle_sampled_tmp = current_angle_original[:, :,
                                    plot_current_data.get_index_at_t_v2(
                                        current_angle_original[0],
                                        x_static_u[2*i][0] + 1000):plot_current_data.get_index_at_t_v2(
                                        current_angle_original[0], x_static_u[2*i+1][-1]) -1000]

        h5f.create_dataset('current_angle_original_' + str(i), data=current_angle_sampled_tmp)
        start_proportion = 0
        end_proportion = 1
        plt = plot_current_data.create_figure((40, 5), 'time', 'current_angle')
        plt = plot_current_data.plot_series_with_time(plt, current_angle_sampled_tmp[0][:,
                                                           int(start_proportion * current_angle_sampled_tmp.shape[
                                                               2]):int(
                                                               end_proportion * current_angle_sampled_tmp.shape[2])],
                                                      "current_angle_x")
        plt = plot_current_data.plot_series_with_time(plt, current_angle_sampled_tmp[1][:,
                                                           int(start_proportion * current_angle_sampled_tmp.shape[
                                                               2]):int(
                                                               end_proportion * current_angle_sampled_tmp.shape[2])],
                                                      "current_angle_y")
        plt.savefig(folder_path + '/' + 'current_angle_' + str(i) + '.png')



        position_from_current_angle_sample_tmp = position_from_current_angle_sample[:, :,
                                    plot_current_data.get_index_at_t_v2(
                                        position_from_current_angle_sample[0],
                                        x_static_u[2 * i][0] + 1000):plot_current_data.get_index_at_t_v2(
                                        position_from_current_angle_sample[0], x_static_u[2 * i + 1][-1]) - 1000]

        start_proportion = 0
        end_proportion = 1
        plt = plot_current_data.create_figure((40, 5), 'time', 'position_from_current_angle_sample_tmp')
        plt.plot(position_from_current_angle_sample_tmp[0][:,int(start_proportion * position_from_current_angle_sample_tmp.shape[2]):int(end_proportion * position_from_current_angle_sample_tmp.shape[2])],position_from_current_angle_sample_tmp[1][:,int(start_proportion * position_from_current_angle_sample_tmp.shape[2]):int(end_proportion * position_from_current_angle_sample_tmp.shape[2])])
        plt.plot(position_from_current_angle_sample_tmp[1][:,
                 int(start_proportion * position_from_current_angle_sample_tmp.shape[2]):int(
                     end_proportion * position_from_current_angle_sample_tmp.shape[2])],
                 position_from_current_angle_sample_tmp[1][:,
                 int(start_proportion * position_from_current_angle_sample_tmp.shape[2]):int(
                     end_proportion * position_from_current_angle_sample_tmp.shape[2])])
        # plt = plot_current_data.plot_series_with_time(plt, position_from_current_angle_sample_tmp[0][:,
        #                                                    int(start_proportion * position_from_current_angle_sample_tmp.shape[
        #                                                        2]):int(
        #                                                        end_proportion * position_from_current_angle_sample_tmp.shape[2])],
        #                                               "position_from_current_angle_sample_tmp_x")
        # plt = plot_current_data.plot_series_with_time(plt, position_from_current_angle_sample_tmp[1][:,
        #                                                    int(start_proportion * position_from_current_angle_sample_tmp.shape[
        #                                                        2]):int(
        #                                                        end_proportion * position_from_current_angle_sample_tmp.shape[2])],
        #                                               "position_from_current_angle_sample_tmp_y")
        plt.savefig(folder_path + '/' + 'position_from_current_angle_sample_tmp_' + str(i) + '.png')
    h5f.close()


if __name__ == '__main__':

    files = ['20231130-XYshift0.03_F500_BRISK-0001','20231130-XYshift0.03_F1500_BRISK-0001']


    for file in files:
        print(file)
        current = plot_current_data.load_pico_uvw_csv_v5(file_path + file + '.csv')
        print("current.shape: ", current.shape)

        current_angle = plot_current_data.calculate_current_angle(current, num_joints)
        current_angle = plot_current_data.clark_angle_to_2pi(current_angle, num_joints)
        print("current_angle.shape: ", current_angle.shape)

        h5f = h5py.File(train_file_path + 'current_angle/' + file.split('.csv')[0] + '_current_angle_original' + '.h5', 'w')
        h5f.create_dataset('current_angle_original', data=current_angle)
        h5f.close()

        coefficients_x = train.load_angle_pair_model(model_file_path, '20230708-X0+-X100+-F500-0001')
        coefficients_y = train.load_angle_pair_model(model_file_path, '20230616-Y0+-Y100-F500-0001')
        current_angle[0] = train.fit_current_angle(current_angle[0], coefficients_x)
        current_angle[1] = train.fit_current_angle(current_angle[1], coefficients_y)

        h5f = h5py.File(train_file_path + 'current_angle/' + file.split('.csv')[0] + '_current_angle' + '.h5', 'w')
        h5f.create_dataset('current_angle', data=current_angle)
        h5f.close()

        position_from_current_angle = plot_current_data.angle_to_position_v4(current_angle, num_joints, pole_pair,
                                                                             screw_lead, [0, 0])
        position_from_current_angle = plot_current_data.position_negate(position_from_current_angle, num_joints)
        print("position_from_current_angle.shape: ", position_from_current_angle.shape)

        h5f = h5py.File(train_file_path + 'position_from_current_angle/' + file.split('.csv')[0] + '_position_from_current_angle' + '.h5', 'w')
        h5f.create_dataset('position_from_current_angle', data=position_from_current_angle)
        # h5f.close()

        position_from_current_angle_sample = plot_current_data.downsample_v2(position_from_current_angle, num_joints, 1)

        h5f = h5py.File(train_file_path + 'position_from_current_angle/' + file.split('.csv')[0] + '_position_from_current_angle_sample' + '.h5', 'w')
        h5f.create_dataset('position_from_current_angle_sample', data=position_from_current_angle_sample)
        h5f.close()

        split_end_part(file.split('.csv')[0])
