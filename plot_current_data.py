import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
import json
import scipy.signal as sg
import finitestate
from scipy.signal import find_peaks

def judge_movement_test(sequence, init_state, window_size, threshold):
    '''
    目前使用这个版本
    :param sequence:
    :param window_size:
    :param threshold:
    :return:
    '''
    #fsm = FiniteStateMachine_v2(start_window_size,end_window_size, threshold)
    # fsm = FiniteStateMachine_v8(init_state, window_size, threshold)
    # fsm = FiniteStateMachine_v9(init_state, window_size, threshold)
    # fsm = FiniteStateMachine_v10(init_state, window_size, threshold)
    # fsm = FiniteStateMachine_v10_noprint(init_state, window_size, threshold)
    fsm = finitestate.FiniteStateMachine_v11(init_state, window_size, threshold)

    movement = []

    t = []
    for i in range(sequence.shape[1]):
        #print("时刻: ",sequence[0][i])
        # start_time = time.perf_counter()
        fsm.update_state([sequence[0][i], sequence[1][i]])
        # end_time = time.perf_counter()
        # elapsed_time_microseconds = (end_time - start_time) * 1e6  # 将秒转换为微秒
        # t.append(elapsed_time_microseconds)
        # print(f"程序执行时间：{elapsed_time_microseconds:.2f} 微秒")
        # print("还原耗时: ", [sequence.shape[1], elapsed_time_microseconds])
        #print("状态: ", fsm.get_state())
        movement.append(fsm.get_state())

    # f = open('./t_consum.txt','a')
    # for i in range(len(t)):
    #     f.write(str(movement[i]) +','+ str(t[i]) + '\n')
    #     # f.write(str(t[i])+'\n')
    # f.close()

    return np.array([sequence[0][window_size:-window_size],movement[2*window_size:]])

def load_csv(file_name, num_joints):
    """
    Load structure with time data from csv file

    :param file_name: file path and file name
    :param num_joints: num of joints
    :returns: a numpy array shape:(num_joints, 2, time_n)
    :raises keyError: raises an exception
    """
    col_list = [f'joint{i}' for i in range(num_joints)]
    col_list = ['time', 'timestamp'] + col_list
    df = pd.read_csv(file_name, encoding='utf8', names=col_list)
    val = []
    for i in range(num_joints):
        val.append(np.array(df[f'joint{i}'].to_list()))
    val = np.array(val)

    time_tmp = np.array(df['timestamp'].to_list())
    time_tmp = time_tmp - time_tmp[0]

    result = []
    for i in range(num_joints):
        result.append([time_tmp, val[i]])
    return np.array(result)

def create_figure(fig_size, xlabel, ylabel):
    plt.figure(figsize=fig_size)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(labelsize=20)
    # plt.legend( prop = {'size':15})
    # plt.axes().get_xaxis().set_visible(False)  # 隐藏x坐标轴
    return plt

def scatter_series_with_time(plt, iterm, iterm_name):
    """
    Plot scatter time series iterm

    :param plt: plt handle
    :param iterm: time series item shape:(2, time_stamp)
    :param iterm_name: iterm_name
    :returns: none
    :raises keyError: raises an exception
    """
    x = iterm[0]
    plt.scatter(x, iterm[1], s=1, label=iterm_name)
    #     plt.ylim(0, 1)
    plt.legend()
    return plt

def plot_series_with_time(plt, iterm, iterm_name):
    """
    Plot time series iterm

    :param plt: plt handle
    :param iterm: time series item shape:(2, time_stamp)
    :param iterm_name: iterm_name
    :returns: none
    :raises keyError: raises an exception
    """
    x = iterm[0]
    plt.plot(x, iterm[1], label=iterm_name)
    #     plt.ylim(0, 1)
    plt.legend()
    return plt

def plot_series(iterm, iterm_name):
    """
    Plot series iterm

    :param file_name: series item shape:(n, )
    :returns: none
    :raises keyError: raises an exception
    """
    x = np.array([i for i in range(iterm.shape[0])])
    plt.plot(x, iterm, label=iterm_name)
    #     plt.ylim(0, 1)
    plt.legend()
    return plt

def integrate_timeseries(iterm):
    """
    Integrate time series iterm

    :param iterm: time series item shape:(2, time_stamp)
    :returns: the result of the integral operation shape:(2, time_stamp)
    :raises keyError: raises an exception
    """
    p = [0]
    for i in range(1, iterm.shape[1]):
        t_delta = iterm[0][i] - iterm[0][i - 1]
        p_delta = iterm[1][i] * t_delta
        p.append(p[-1] + p_delta)
    return np.array([iterm[0], np.array(p)])

def downsampling(iterm, rate):
    """
    Downsampling a timeseries.已经不用了

    :param iterm: timeseries shape:(2, timestamp)
    :param rate: downsampling rate
    :returns: downsampled timeseries shape:(2,timestamp)
    :raises keyError: raises an exception
    """
    t = []
    val = []
    for i in range(iterm.shape[1]):
        if i % rate == 0:
            t.append(iterm[0][i])
            val.append(iterm[1][i])
    t = np.array(t)
    val = np.array(val)
    iterm_resample = np.array([t, val])
    return iterm_resample

def position_radian_to_mm(iterm, num_joints, screw_lead):
    """
    Convert radian(rad) to position(mm)

    :param iterm: timeseries shape:(num_joints, 2, timestamp)
    :param num_joints: num of joints
    :param screw_lead: screw lead length of system
    :returns: position in mm%system
    :raises keyError: raises an exception
    """
    result = []
    for i in range(num_joints):
        time_tmp = iterm[i][0]
        result.append([time_tmp, (iterm[i][1]) / (2 * np.pi) * screw_lead])
    return np.array(result)

def position_mm_to_e_angle(iterm, num_joints, screw_lead, pole_pair):
    """
    Convert position(mm) to e_angle

    :param iterm: timeseries shape:(num_joints, 2, ,timestamp)
    :param num_joints: num of joints
    :param pole_pair: pole_pair of system
    :returns: e_angle
    :raises keyError: raises an exception
    """
    result = []
    for i in range(num_joints):
        time_tmp = iterm[i][0]
        result.append([time_tmp, ((iterm[i][1] / screw_lead * (2 * np.pi)) % (2 * np.pi) * pole_pair) % (2 * np.pi)])
    return np.array(result)

def clark_angle_to_2pi(iterm, num_joints):
    """
    将-pi到pi的clark_angle转换到0到2pi
    """
    result = []
    for i in range(num_joints):
        time_tmp = iterm[i][0]
        tmp = []
        for j in range(iterm[i].shape[1]):
            if iterm[i][1][j] < 0:
                tmp.append(iterm[i][1][j] + 2 * np.pi)
            else:
                tmp.append(iterm[i][1][j])
        result.append([time_tmp, np.array(tmp)])
    return np.array(result)

def find_start(file_path, nc_filename, num_joints):
    cnc_vel_cmd = load_csv(file_path + nc_filename + '-joint_log_vel_cmd.csv', num_joints)
    for i in range(cnc_vel_cmd.shape[2]):
        if cnc_vel_cmd[0][1][i] != 0:
            return i, cnc_vel_cmd[0][0][i]

def sliding_window(iterm, num_joints, step):
    """
    输入(3,2,time_stamp)
    输出(3,t(ms),2,n(每tms内包含的点))
    """
    r = []
    for joint in range(num_joints):
        t = step
        tmp_t = []
        tmp_value = []
        result = []
        for i in range(iterm.shape[2]):
            if iterm[0][0][i] <= t:
                tmp_t.append(iterm[joint][0][i])
                tmp_value.append(iterm[joint][1][i])
            else:
                tmp_t = np.array(tmp_t)
                tmp_value = np.array(tmp_value)
                result.append(np.array([tmp_t, tmp_value]))
                tmp_t = []
                tmp_value = []
                tmp_t.append(iterm[joint][0][i])
                tmp_value.append(iterm[joint][1][i])
                t = t + step
        if tmp_t:
            tmp_t = np.array(tmp_t)
            tmp_value = np.array(tmp_value)
            result.append(np.array([tmp_t, tmp_value]))
        r.append(result)
    return r

def angle_to_position(iterm, num_joints):
    """
    将0到2pi的角度转换为位置轨迹
    (电角度轨迹)
    """
    position_from_e_angle = []
    for joint in range(num_joints):
        iterm_re = iterm[joint][1][0]  # 设置起始位置
        p = [iterm_re]
        incre = 0
        for i in range(iterm.shape[2] - 1):
            #             print("\r", end="")
            #             print("Angle_to_position progress: {}%: ".format(i/(iterm.shape[2]-1)), "▋" * (i // ((iterm.shape[2]-1)//50)), end="")
            #             sys.stdout.flush()
            if abs((incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i])) > 6:
                if (incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i]) > 0:
                    incre = -math.ceil(((incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i])) / (2 * np.pi))
                    iterm_re = iterm_re + (incre * 2 * np.pi + iterm[joint][1][i + 1]) - (iterm[joint][1][i])
                    p.append(iterm_re)
                else:
                    incre = math.ceil(
                        (-((incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i]))) / (2 * np.pi))
                    iterm_re = iterm_re + (incre * 2 * np.pi + iterm[joint][1][i + 1]) - (iterm[joint][1][i])
                    p.append(iterm_re)
            else:
                iterm_re = iterm_re + (incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i])
                p.append(iterm_re)
        p = np.array(p)
        position_from_e_angle.append(np.array([iterm[joint][0], p]))
    return np.array(position_from_e_angle)

def angle_to_position_v2(iterm, num_joints, pole_pair):
    """
    将0到2pi的角度转换为位置轨迹
    （机械角度轨迹）
    """
    position_from_e_angle = []
    for joint in range(num_joints):
        iterm_re = iterm[joint][1][0]  # 设置起始位置
        p = [iterm_re]
        incre = 0
        for i in range(iterm.shape[2] - 1):
            #             print("\r", end="")
            #             print("Angle_to_position progress: {}%: ".format(i/(iterm.shape[2]-1)), "▋" * (i // ((iterm.shape[2]-1)//50)), end="")
            #             sys.stdout.flush()
            if abs((incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i])) > 6.2:
                #                 print(i)
                if (incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i]) > 0:
                    incre = -math.ceil(((incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i])) / (2 * np.pi))
                    iterm_re = iterm_re + (incre * 2 * np.pi + iterm[joint][1][i + 1]) - (iterm[joint][1][i])
                    p.append(iterm_re)
                else:
                    incre = math.ceil(
                        (-((incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i]))) / (2 * np.pi))
                    iterm_re = iterm_re + (incre * 2 * np.pi + iterm[joint][1][i + 1]) - (iterm[joint][1][i])
                    p.append(iterm_re)
            else:
                iterm_re = iterm_re + (incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i])
                p.append(iterm_re)
        p = np.array(p)
        start_point = p[0]
        p = p / pole_pair
        p = p + (start_point - p[0])
        position_from_e_angle.append(np.array([iterm[joint][0], p]))
    return np.array(position_from_e_angle)

def angle_to_position_v3(iterm, num_joints, pole_pair, screw_lead, start_point):
    """
    将0到2pi的角度转换为位置轨迹
    （运动轨迹mm）
    """
    position_from_e_angle = []
    for joint in range(num_joints):
        iterm_re = iterm[joint][1][0]  # 设置起始位置
        p = [iterm_re]
        incre = 0
        for i in range(iterm.shape[2] - 1):
            #             print("\r", end="")
            #             print("Angle_to_position progress: {}%: ".format(i/(iterm.shape[2]-1)), "▋" * (i // ((iterm.shape[2]-1)//50)), end="")
            #             sys.stdout.flush()
            if abs((incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i])) > 6.2:
                #                 print(i)
                if (incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i]) > 0:
                    incre = -math.ceil(((incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i])) / (2 * np.pi))
                    iterm_re = iterm_re + (incre * 2 * np.pi + iterm[joint][1][i + 1]) - (iterm[joint][1][i])
                    p.append(iterm_re)
                else:
                    incre = math.ceil(
                        (-((incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i]))) / (2 * np.pi))
                    iterm_re = iterm_re + (incre * 2 * np.pi + iterm[joint][1][i + 1]) - (iterm[joint][1][i])
                    p.append(iterm_re)
            else:
                iterm_re = iterm_re + (incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i])
                p.append(iterm_re)
        p = np.array(p)
        p = p / pole_pair
        p = p / (2 * np.pi) * screw_lead
        p = p + (start_point - p[0])
        position_from_e_angle.append(np.array([iterm[joint][0], p]))
    return np.array(position_from_e_angle)

def angle_to_position_v4(iterm, num_joints, pole_pair, screw_lead, start_point):
    """
    将0到2pi的角度转换为位置轨迹，不做修正时使用的版本
    （运动轨迹mm）
    """
    position_from_e_angle = []
    for joint in range(num_joints):
        iterm_re = iterm[joint][1][0]  # 设置起始位置
        #iterm_re = 100
        #iterm_re = np.mean(iterm[joint][1][0:3000])  # 设置起始位置
        #print("qishiweizhi: ",iterm_re)
        p = [iterm_re]
        incre = 0
        for i in range(iterm.shape[2] - 1):
            #             print("\r", end="")
            #             print("Angle_to_position progress: {}%: ".format(i/(iterm.shape[2]-1)), "▋" * (i // ((iterm.shape[2]-1)//50)), end="")
            #             sys.stdout.flush()
            if abs((incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i])) > 3.1:#6.2
                #                 print(i)
                if (incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i]) > 0:
                    incre = -math.ceil(((incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i])) / (2 * np.pi))
                    iterm_re = iterm_re + (incre * 2 * np.pi + iterm[joint][1][i + 1]) - (iterm[joint][1][i])
                    p.append(iterm_re)
                else:
                    incre = math.ceil(
                        (-((incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i]))) / (2 * np.pi))
                    iterm_re = iterm_re + (incre * 2 * np.pi + iterm[joint][1][i + 1]) - (iterm[joint][1][i])
                    p.append(iterm_re)
            else:
                iterm_re = iterm_re + (incre + iterm[joint][1][i + 1]) - (incre + iterm[joint][1][i])
                p.append(iterm_re)
        p = np.array(p)
        p = p / pole_pair
        p = p / (2 * np.pi) * screw_lead
        #p = p + (start_point[joint] - p[0])
        p = p + (start_point[joint] - np.mean(p[0:1000]))
        position_from_e_angle.append(np.array([iterm[joint][0], p]))
        #print('flag')
    return np.array(position_from_e_angle)

def find_end_offset(re,tar):
    tar = round(tar % 2.5,3)
    print("tar: ",tar)
    for i in range(len(re)):
        if re[i][0] >= tar:
            return (re[i][2]-re[i][1])/(2*np.pi) * 2.5
    return 0

def find_s2e_offset(re,tar,tag):
    '''
    :param re:
    :param tar:
    :param tag: 0 代表current_angle下降，1代表current_angle上升
    :return:
    '''
    tar = round(tar % 2.5, 3)
    print("tar: ", tar)
    for i in range(len(re)):
        if re[i][0] >= tar:
            if tag == 0:
                if re[i][2] < re[i][1]:
                    return -1 * (re[i][2]-re[i][1])/(2*np.pi) * 2.5
                else:
                    return 1 * (2 * np.pi - re[i][2] + re[i][1]) / (2 * np.pi) * 2.5
            else:
                if re[i][2] > re[i][1]:
                    return -1 * (re[i][2]-re[i][1])/(2*np.pi) * 2.5
                else:
                    return -1 * (2 * np.pi + re[i][2] - re[i][1]) / (2 * np.pi) * 2.5
    return 0

def position_fix_v2(position_from_current_angle_sampled_smoothed, movement, num_joints, window_size, start_point, end_forward, end_reverse,reverse2forward,forward2reverse):
    """
    将0到2pi的角度转换为位置轨迹，FiniteStateMachine_v6使用的版本
    （运动轨迹mm）
    """
    p = []
    for joint in range(num_joints):
        position_fix = [-1*start_point[joint]]
        offset = 0
        for i in range(1, len(movement[0][0])):
            if (movement[joint][1][i][0] == 0) | (movement[joint][1][i][0] == 1):
                position_fix.append(position_fix[-1])
            elif (movement[joint][1][i][0] == 2):
                if (movement[joint][1][i-1][0] == 0):
                    offset = 0
                    offset = offset + find_end_offset(end_forward[joint],abs(position_fix[-1])) + position_from_current_angle_sampled_smoothed[joint][1][window_size+i] - position_fix[-1]#这里改了
                    #position_fix.append(position_fix[-1] - offset)
                    position_fix.append(position_from_current_angle_sampled_smoothed[joint][1][window_size + i] - offset)
                elif (movement[joint][1][i - 1][0] == 2):
                    position_fix.append(position_from_current_angle_sampled_smoothed[joint][1][window_size+i] - offset)
                elif (movement[joint][1][i - 1][0] == 4):
                    offset = 0
                    offset = offset + position_from_current_angle_sampled_smoothed[joint][1][window_size+i] - position_fix[-1]
                    position_fix.append(position_from_current_angle_sampled_smoothed[joint][1][window_size+i] - offset)
                elif (movement[joint][1][i - 1][0] == 1):
                    if movement[joint][1][i][1] == 0:
                        offset = 0
                        offset = offset + find_s2e_offset(reverse2forward[joint], abs(position_fix[-1]), 1) + position_from_current_angle_sampled_smoothed[joint][1][window_size + i] - position_fix[-1]
                        position_fix.append(position_from_current_angle_sampled_smoothed[joint][1][window_size + i] - offset)
                    elif movement[joint][1][i][1] == 1:
                        offset = 0
                        offset = offset + find_s2e_offset(reverse2forward[joint], abs(position_fix[-1]), 0) + position_from_current_angle_sampled_smoothed[joint][1][window_size + i] - position_fix[-1]
                        position_fix.append(position_from_current_angle_sampled_smoothed[joint][1][window_size + i] - offset)
            elif (movement[joint][1][i][0] == 3):
                if (movement[joint][1][i-1][0] == 1):
                    offset = 0
                    offset = offset + find_end_offset(end_reverse[joint],abs(position_fix[-1])) + position_from_current_angle_sampled_smoothed[joint][1][window_size+i] - position_fix[-1]#这里改了
                    #position_fix.append(position_fix[-1] - offset)
                    position_fix.append(position_from_current_angle_sampled_smoothed[joint][1][window_size + i] - offset)
                elif (movement[joint][1][i - 1][0] == 3):
                    position_fix.append(position_from_current_angle_sampled_smoothed[joint][1][window_size+i] - offset)
                elif (movement[joint][1][i - 1][0] == 4):
                    offset = 0
                    offset = offset + position_from_current_angle_sampled_smoothed[joint][1][window_size + i] - position_fix[-1]
                    position_fix.append(position_from_current_angle_sampled_smoothed[joint][1][window_size + i] - offset)
                elif (movement[joint][1][i - 1][0] == 0):
                    if movement[joint][1][i][1] == 0:
                        offset = 0
                        offset = offset + find_s2e_offset(forward2reverse[joint], abs(position_fix[-1]), 0) + position_from_current_angle_sampled_smoothed[joint][1][window_size + i] - position_fix[-1]
                        position_fix.append(position_from_current_angle_sampled_smoothed[joint][1][window_size + i] - offset)
                    elif movement[joint][1][i][1] == 1:
                        offset = 0
                        offset = offset + find_s2e_offset(forward2reverse[joint], abs(position_fix[-1]), 1) + position_from_current_angle_sampled_smoothed[joint][1][window_size + i] - position_fix[-1]
                        position_fix.append(position_from_current_angle_sampled_smoothed[joint][1][window_size + i] - offset)
            elif (movement[joint][1][i][0] == 4):
                position_fix.append(position_fix[-1])
        p.append([movement[joint][0],position_fix])
    return np.array(p)

def position_fix_v3(position_from_current_angle_sampled_smoothed, joint, t, state, end_forward, end_reverse,reverse2forward,forward2reverse):
    """
    修复t时刻的偏移
    与白名单匹配版本，不在白名单中的状态转移将不进行修正，将0到2pi的角度转换为位置轨迹，FiniteStateMachine_v6使用的版本
    （运动轨迹mm）
    """

    index = get_index_at_t_v2(position_from_current_angle_sampled_smoothed[joint], t)
    if (state[1][0] == 2):
        if (state[0][0] == 0):
            # print('0到2')
            # print("启动位置: ", position_from_current_angle_sampled_smoothed[joint][1][index - 3])
            # print("启动偏移: ",find_end_offset(end_forward[joint],abs(position_from_current_angle_sampled_smoothed[joint][1][index - 3])))
            offset = find_end_offset(end_forward[joint], abs(position_from_current_angle_sampled_smoothed[joint][1][index - 3])) + position_from_current_angle_sampled_smoothed[joint][1][index - 2] - position_from_current_angle_sampled_smoothed[joint][1][index - 3]
            # print("启动偏移: ", offset)
            position_from_current_angle_sampled_smoothed[joint][1][index - 2:] = position_from_current_angle_sampled_smoothed[joint][1][index - 2:] - offset
        elif (state[0][0] == 4):
            # print('4到2')
            # print("启动位置: ", position_from_current_angle_sampled_smoothed[joint][1][index - 3])
            offset = position_from_current_angle_sampled_smoothed[joint][1][index - 2] - position_from_current_angle_sampled_smoothed[joint][1][index - 3]
            # print("启动偏移: ", offset)
            position_from_current_angle_sampled_smoothed[joint][1][index - 2:] = position_from_current_angle_sampled_smoothed[joint][1][index - 2:] - offset

        elif (state[0][0] == 1):
            # print('1到2')
            # print("启动位置: ", position_from_current_angle_sampled_smoothed[joint][1][index - 3])
            if state[1][1] == 0:
                # print("启动偏移: ", find_s2e_offset(reverse2forward[joint], abs(position_from_current_angle_sampled_smoothed[joint][1][index - 3]), 1))
                offset = find_s2e_offset(reverse2forward[joint], abs(position_from_current_angle_sampled_smoothed[joint][1][index - 3]), 1) + position_from_current_angle_sampled_smoothed[joint][1][index - 2] - position_from_current_angle_sampled_smoothed[joint][1][index - 3]
                # print("启动偏移: ", offset)
                position_from_current_angle_sampled_smoothed[joint][1][index - 2:] = position_from_current_angle_sampled_smoothed[joint][1][index - 2:] - offset
            elif state[1][1] == 1:
                # print("启动偏移: ", find_s2e_offset(reverse2forward[joint], abs(position_from_current_angle_sampled_smoothed[joint][1][index - 3]), 0))
                offset =  find_s2e_offset(reverse2forward[joint], abs(position_from_current_angle_sampled_smoothed[joint][1][index - 3]), 0) + position_from_current_angle_sampled_smoothed[joint][1][index - 2] - position_from_current_angle_sampled_smoothed[joint][1][index - 3]
                # print("启动偏移: ", offset)
                position_from_current_angle_sampled_smoothed[joint][1][index - 2:] = position_from_current_angle_sampled_smoothed[joint][1][index - 2:] - offset
    elif (state[1][0] == 3):
        if (state[0][0] == 1):
            # print('1到3')
            # print("启动位置: ", position_from_current_angle_sampled_smoothed[joint][1][index - 3])
            # print("启动偏移: ",find_end_offset(end_reverse[joint],abs(position_from_current_angle_sampled_smoothed[joint][1][index - 3])))
            offset =  find_end_offset(end_reverse[joint],abs(position_from_current_angle_sampled_smoothed[joint][1][index - 3])) + position_from_current_angle_sampled_smoothed[joint][1][index - 2] - position_from_current_angle_sampled_smoothed[joint][1][index - 3]
            # print("启动偏移: ", offset)
            position_from_current_angle_sampled_smoothed[joint][1][index - 2:] = position_from_current_angle_sampled_smoothed[joint][1][index - 2:] - offset
        elif (state[0][0] == 4):
            # print('4到3')
            offset = position_from_current_angle_sampled_smoothed[joint][1][index - 2] - position_from_current_angle_sampled_smoothed[joint][1][index - 3]
            # print("启动偏移: ", offset)
            position_from_current_angle_sampled_smoothed[joint][1][index - 2:] = position_from_current_angle_sampled_smoothed[joint][1][index - 2:] - offset
        elif (state[0][0] == 0):
            # print('0到3')
            # print("启动位置: ", position_from_current_angle_sampled_smoothed[joint][1][index - 3])
            if state[1][1] == 0:
                # print("启动偏移: ", find_s2e_offset(forward2reverse[joint], abs(position_from_current_angle_sampled_smoothed[joint][1][index - 3]), 0))
                offset = find_s2e_offset(forward2reverse[joint], abs(position_from_current_angle_sampled_smoothed[joint][1][index - 3]), 0) + position_from_current_angle_sampled_smoothed[joint][1][index - 2] - position_from_current_angle_sampled_smoothed[joint][1][index - 3]
                # print("启动偏移: ", offset)
                position_from_current_angle_sampled_smoothed[joint][1][index - 2:] = position_from_current_angle_sampled_smoothed[joint][1][index - 2:] - offset
            elif state[1][1] == 1:
                # print("启动偏移: ", find_s2e_offset(forward2reverse[joint], abs(position_from_current_angle_sampled_smoothed[joint][1][index - 3]), 1))
                offset =  find_s2e_offset(forward2reverse[joint], abs(position_from_current_angle_sampled_smoothed[joint][1][index - 3]), 1) + position_from_current_angle_sampled_smoothed[joint][1][index - 2] - position_from_current_angle_sampled_smoothed[joint][1][index - 3]
                # print("启动偏移: ", offset)
                position_from_current_angle_sampled_smoothed[joint][1][index - 2:] = position_from_current_angle_sampled_smoothed[joint][1][index - 2:] - offset
    # elif (state[1][0] == 0) | (state[1][0] == 1):
    #     # position_from_current_angle_sampled_smoothed[joint][1][index:] = position_from_current_angle_sampled_smoothed[joint][1][index]
    #     position_from_current_angle_sampled_smoothed[joint][1][index - 2:] = position_from_current_angle_sampled_smoothed[joint][1][index]
    return position_from_current_angle_sampled_smoothed[joint]


def position_fix_v0(position_from_current_angle_sampled_smoothed, movement, num_joints, window_size, start_point, end_forward, end_reverse):
    """
    将0到2pi的角度转换为位置轨迹，带运动状态机使用的版本
    （运动轨迹mm）
    """
    p = []
    for joint in range(num_joints):
        position_fix = [-1*start_point[joint]]
        offset = 0
        for i in range(1, movement.shape[2]):
            if (movement[joint][1][i] == 0) | (movement[joint][1][i] == 1):
                position_fix.append(position_fix[-1])
            elif (movement[joint][1][i] == 2):
                if (movement[joint][1][i-1] == 0):
                    offset = 0
                    print("启动位置: ", position_fix[-1])
                    print("启动偏移: ",find_end_offset(end_forward[joint],abs(position_fix[-1])))
                    offset = offset + find_end_offset(end_forward[joint],abs(position_fix[-1]))
                    position_fix.append(position_fix[-1] - offset)
                elif (movement[joint][1][i - 1] == 2):
                    position_fix.append(position_from_current_angle_sampled_smoothed[joint][1][window_size+i] - offset)
                elif (movement[joint][1][i - 1] == 4):
                    offset = 0
                    print('pianyi')
                    print(position_from_current_angle_sampled_smoothed[joint][0][window_size + i])
                    print(position_from_current_angle_sampled_smoothed[joint][1][window_size+i])
                    print(position_fix[-1])
                    offset = offset + position_from_current_angle_sampled_smoothed[joint][1][window_size+i] - position_fix[-1]
                    position_fix.append(position_from_current_angle_sampled_smoothed[joint][1][window_size+i] - offset)
            elif (movement[joint][1][i] == 3):
                if (movement[joint][1][i-1] == 1):
                    offset = 0
                    print("启动位置: ", position_fix[-1])
                    print("启动偏移: ",find_end_offset(end_reverse[joint],abs(position_fix[-1])))
                    offset = offset + find_end_offset(end_reverse[joint],abs(position_fix[-1]))#这里改了
                    position_fix.append(position_fix[-1] - offset)
                elif (movement[joint][1][i - 1] == 3):
                    position_fix.append(position_from_current_angle_sampled_smoothed[joint][1][window_size+i] - offset)
                elif (movement[joint][1][i - 1] == 4):
                    offset = 0
                    offset = offset + position_from_current_angle_sampled_smoothed[joint][1][window_size + i] - position_fix[-1]
                    position_fix.append(position_from_current_angle_sampled_smoothed[joint][1][window_size + i] - offset)
                # elif (movement[joint][1][i - 1] == 0):#临时用而已，是不对的
                #     position_fix.append(position_from_current_angle_sampled_smoothed[joint][1][window_size+i] - offset)
            elif (movement[joint][1][i] == 4):
                position_fix.append(position_fix[-1])
        p.append([movement[joint][0],position_fix])
    return np.array(p)

def downsample(iterm, num_joints, sample_t):
    """
    每隔sample_t周期，采样一次；如sample为1，即1毫秒采样一次，注意这是最新的，其余的版本可能没有更新，这个版本是每隔1毫秒取一次，剩余的丢掉
    """
    result = []
    for joint in range(num_joints):
        t = 1
        tmp_t = []
        tmp_value = []
        tmp_t.append(iterm[joint][0][0])
        tmp_value.append(iterm[joint][1][0])
        for i in range(iterm.shape[2] - 1):
            #             print('---------------------')
            #             print(iterm[joint][0][i+1])
            #             print(iterm[joint][0][i])
            #             print(iterm[joint][0][0] + t*sample_t)

            next_t = round(iterm[joint][0][i + 1], 8)
            current_t = round(iterm[joint][0][i], 8)
            tar_t = round(iterm[joint][0][0] + t * sample_t, 8)
            # print(next_t)
            # print(current_t)
            # print(tar_t)

            if (next_t >= tar_t) & (current_t < tar_t):
                if abs(next_t - (tar_t)) < abs(current_t - (tar_t)):
                    tmp_t.append(iterm[joint][0][i + 1])
                    tmp_value.append(iterm[joint][1][i + 1])
                else:
                    tmp_t.append(iterm[joint][0][i])
                    tmp_value.append(iterm[joint][1][i])
                t = t + 1
            elif current_t > tar_t:
                tmp_t.append(iterm[joint][0][i])
                tmp_value.append(iterm[joint][1][i])
                t = t + 1
        tmp_t = np.array(tmp_t)
        tmp_value = np.array(tmp_value)
        result.append([tmp_t, tmp_value])
    return np.array(result)

def downsample_v2(iterm, num_joints, sample_t):
    """
    每隔sample_t周期，采样一次；如sample为1，即1毫秒采样一次，注意这是最新的，其余的版本可能没有更新，这个版本是取一毫秒内的平均值
    """
    result = []
    for joint in range(num_joints):
        t = 1
        tmp_t = []
        tmp_value = []
        v_ = []
        #tmp_t.append(iterm[joint][0][0])
        #tmp_value.append(iterm[joint][1][0])
        for i in range(iterm.shape[2] - 1):
            #             print('---------------------')
            #             print(iterm[joint][0][i+1])
            #             print(iterm[joint][0][i])
            #             print(iterm[joint][0][0] + t*sample_t)

            next_t = round(iterm[joint][0][i + 1], 8)
            current_t = round(iterm[joint][0][i], 8)
            tar_t = round(iterm[joint][0][0] + t * sample_t, 8)
            # print(next_t)
            # print(current_t)
            # print(tar_t)

            if (next_t >= tar_t) & (current_t < tar_t):
                if abs(next_t - (tar_t)) < abs(current_t - (tar_t)):
                    tmp_t.append(iterm[joint][0][i + 1])
                    v_.append(iterm[joint][1][i])
                    #tmp_value.append(iterm[joint][1][i + 1])
                    tmp_value.append(np.mean(np.array(v_)))
                else:
                    tmp_t.append(iterm[joint][0][i])
                    v_.append(iterm[joint][1][i])
                    #tmp_value.append(iterm[joint][1][i])
                    tmp_value.append(np.mean(np.array(v_)))
                t = t + 1
                v_ = []
            elif current_t > tar_t:
                tmp_t.append(iterm[joint][0][i])
                v_.append(iterm[joint][1][i])
                #tmp_value.append(iterm[joint][1][i])
                tmp_value.append(np.mean(np.array(v_)))
                t = t + 1
                v_ = []
            else:
                v_.append(iterm[joint][1][i])
        tmp_t = np.array(tmp_t)
        tmp_value = np.array(tmp_value)
        result.append([tmp_t, tmp_value])
    return np.array(result)

def calculate_tracking_error(ref_pos, pos, num_joints):
    """
    计算ref_pos与pos之间的跟踪误差
    """
    t = 0
    if ref_pos.shape[2] > pos.shape[2]:
        t = pos.shape[2]
    else:
        t = ref_pos.shape[2]

    result = []
    for joint in range(num_joints):
        tr = []
        for i in range(t):
            tr.append(ref_pos[joint][1][i] - pos[joint][1][i])
        tr = np.array(tr)
        result.append([pos[joint][0][:t], tr])
    result = np.array(result)
    return result

def position_translate(iterm, offset, num_joints):
    """
    将iterm mm整体偏移offset mm
    """
    result = []
    for i in range(num_joints):
        result.append([iterm[i][0], iterm[i][1] + offset])
    return np.array(result)

def load_pico_uvw_csv_v3(file_name):
    """
    load pico csv, [t,u,v,w],pico转成csv后可直接处理,t的单位为毫秒

    :param file_name: pico csv filename
    :returns: 3 * 2 * timestamps
    :raises keyError: raises an exception
    """
    df = pd.read_csv(file_name)

    df = df.drop(labels=0)
    data = df.values.astype(float)

    # print(type(data))
    # print(data)
    data = data.transpose(1, 0)
    t = data[0, :]
    t = t*1000 #t的单位为毫秒
    t = t - t[0]
    current = data[1:4, :]

    channel_u = np.array([t, current[0]])
    channel_v = np.array([t, current[1]])
    channel_w = np.array([t, current[2]])

    return np.array([channel_u, channel_v, channel_w])

def load_pico_uvw_csv_v4(file_name):
    """
    load pico csv, [t,u,v], calculate w phase current via 0-u-v，pico转成csv后可直接处理

    :param file_name: pico csv filename
    :returns: 3 * 2 * timestamps
    :raises keyError: raises an exception
    """
    df = pd.read_csv(file_name)

    df = df.drop(labels=0)
    data = df.values.astype(float)
    # print(type(data))
    # print(data)
    data = data.transpose(1, 0)
    t = data[0, :]
    t = t * 1000  # t的单位为毫秒
    t = t - t[0]
    current = data[1:3, :]

    channel_u = np.array([t, current[0]])
    channel_v = np.array([t, current[1]])
    current_w = 0 - current[0] - current[1]
    channel_w = np.array([t, current_w])

    return np.array([channel_u, channel_v, channel_w])

def load_pico_uvw_csv_v5(file_name):
    """
    load pico csv, [t,u,v,u,v], calculate w phase current via 0-u-v，pico转成csv后可直接处理,t的单位是毫秒ms

    :param file_name: pico csv filename
    :returns: 2(xy) * 3 * 2 * timestamps
    :raises keyError: raises an exception
    """
    df = pd.read_csv(file_name)

    df = df.drop(labels=0)
    data = df.values.astype(float)
    # print(type(data))
    # print(data.shape)
    # print(data[0])
    # print(data[1])
    data = data.transpose(1, 0)
    t = data[0, :]
    t = t * 1000  # t的单位为毫秒
    t = t - t[0]
    current1 = data[1:3, :]

    channel1_u = np.array([t, current1[0]])
    channel1_v = np.array([t, current1[1]])
    current1_w = 0 - current1[0] - current1[1]
    channel1_w = np.array([t, current1_w])

    current2 = data[3:5, :]

    channel2_u = np.array([t, current2[0]])
    channel2_v = np.array([t, current2[1]])
    current2_w = 0 - current2[0] - current2[1]
    channel2_w = np.array([t, current2_w])

    return np.array([[channel1_u, channel1_v, channel1_w],[channel2_u, channel2_v, channel2_w]])

def calculate_current_angle(iterm, num_joints):
    """
    calculate current angle

    :param iterm: 3(num_joints) * 3(channel) * 2 * timestamps
    :param num_joints: num of joints
    :returns: current angle shape: 3(num_joints) * 2 * timestamps
    :raises keyError: raises an exception
    """

    M = np.array([[1, -(1 / 2), -(1 / 2)], [0, math.sqrt(3) / 2, - math.sqrt(3) / 2]])
    M = math.sqrt(2 / 3) * M

    result = []
    for joint in range(num_joints):
        t = iterm[joint][0][0]
        current = iterm[joint][:, 1, :]

        clark = np.dot(M, current)

        angle = []
        for i in range(clark.shape[1]):
            angle.append(math.atan2(clark[0][i], clark[1][i]))

        angle = np.array([t, angle])
        result.append(angle)
    return np.array(result)

def position_negate(iterm, num_joints):
    """
    calculate negate position

    :param iterm: 3(num_joints) * 2 * timestamps
    :param num_joints: num of joints
    :returns: negate position shape: 3(num_joints) * 2 * timestamps
    :raises keyError: raises an exception
    """
    result = []
    for joint in range(num_joints):
        tmp = -1 * iterm[joint][1]
        result.append(np.array([iterm[joint][0], tmp]))
    return np.array(result)

def load_ruler(path, file_name):
    """
    load Grating ruler measurements

    :param file_name: measurements file name json
    :returns: shape: 3(num_joints) * 2 * timestamps
    :raises keyError: raises an exception
    """
    # 打开文件,r是读取,encoding是指定编码格式
    with open(path + file_name + '.json', 'r', encoding='utf-8') as fp:
        data = json.load(fp)  # 输出结果是 <class 'dict'> 一个python对象,json模块会根据文件类对象自动转为最符合的数据类型,所以这里是dict
    fp.close()

    keys = list(data.keys())
    keys.remove('Fs')

    result = []
    for i in keys:
        pos = np.array(data[i])
        t = []
        for i in range(pos.shape[0]):
            t.append(i * (1.0 / data['Fs']))
        t = np.array(t)
        pos = np.array([t, pos])
        result.append(pos)

    return np.array(result)

def find_white_anchor_point(iterm, num_joints):
    result = []
    for joint in range(num_joints):
        anchor = []
        stop = []
        start = []
        reverse = []
        for i in range(iterm.shape[2] - 1):
            if iterm[joint][1][i] * iterm[joint][1][i + 1] == 0:
                if iterm[joint][1][i] != 0:
                    stop.append(i)
                    anchor.append(i)
                elif iterm[joint][1][i + 1] != 0:
                    start.append(i + 1)
                    anchor.append(i + 1)
            elif iterm[joint][1][i] * iterm[joint][1][i + 1] < 0:
                reverse.append(i + 1)
                anchor.append(i + 1)
        start = np.array(start)
        stop = np.array(stop)
        reverse = np.array(reverse)
        anchor = np.array(anchor)
        result.append([start, stop, reverse, anchor])
    return result

def plot_traj(plt,position_from_current_angle,position_from_current_angle_sampled_smoothed,position_from_current_angle_sampled_smoothed_compensate,h_only_sample_points):
    #plt = create_figure((20,5),'time','traj')
    #plt.title(title)
    # plt.plot(position_from_current_angle[0,1,:],position_from_current_angle[1,1,:],label = 'current_traj')
    # plt.plot(position_from_current_angle_sampled_smoothed[0, 1, :], position_from_current_angle_sampled_smoothed[1, 1, :], label='current_traj_smoothed')
    plt.scatter(position_from_current_angle_sampled_smoothed_compensate[0, 1, :],
             position_from_current_angle_sampled_smoothed_compensate[1, 1, :], label='current_traj_smoothed_fsm')
    # plt.plot(h_only_sample_points[0, 1, :], h_only_sample_points[1, 1, :], label='opc_traj')
    # plt.scatter(h_only_sample_points[0, 1, :], h_only_sample_points[1, 1, :], label='opc_traj')
    plt.legend()
    #plt.show()
    return plt

def e_angle_to_m_angle(iterm, pole_pair,num_joints):
    m_angle = []
    for joint in range(num_joints):
        p = iterm[joint][1]
        p = p / pole_pair
        m_angle.append(np.array([iterm[joint][0], p]))
    return np.array(m_angle)

def binarySearchUpperBound2(A, target):
    if target > A[-1]:
        return -1
    low, high = 0, len(A) - 1
    while low <= high:
        mid = low + (high - low) // 2
        if target >= A[mid]:
            low = mid + 1
        else:
            high = mid - 1
    if high < len(A):
        return high + 1
    else:
        return -1


# 函数名应该改为get_value_at_t
def get_position_at_t_v2(iterm, t):
    index = binarySearchUpperBound2(iterm[0], t)
    if index == -1:
        return 0
    else:
        return iterm[1][index]

# 函数名应该改为get_value_at_t
def get_index_at_t_v2(iterm, t):
    index = binarySearchUpperBound2(iterm[0], t)
    if index == -1:
        return 0
    else:
        return index

def moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, mode='same')
    return smoothed_data


def get_maxima(values: np.ndarray):
    """极大值点"""
    max_index = sg.argrelmax(values)[0]
    return [max_index, values[max_index]]

def get_minima(values: np.ndarray):
    """极小值点"""
    min_index = sg.argrelmin(values)[0]  # 极小值的下标
    return [min_index, values[min_index]]  # 返回极小值

if __name__ == '__main__':
    print("test")