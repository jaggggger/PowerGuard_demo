# This is a sample Python script.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plot_current_data
import gcode_interpreter
from scipy.spatial import KDTree
plane = 1#1为XY平面；2为YZ平面；3为XZ平面


def load_csv_v3(file_name, num_joints):
    """
    Load structure with time data from csv file，有三个轴的actToolpos数值，还有三个编码器读数，还有三个轴的速度

    :param file_name: file path and file name
    :param num_joints: num of joints
    :returns: a numpy array shape:(num_joints, 2, time_n)；and a numpy array shape:(num_joints, 2, time_n)
    :raises keyError: raises an exception
    """
    col_list = ['PrimaryKey', 'DataType', 'Value', 'StatusCode', 'SourceTimeStamp', 'ServerTimeStamp']
    df = pd.read_csv(file_name, encoding='utf8', names=col_list)
    val = []
    t = []
    for i in range(num_joints):
        val.append((list(map(float, df[df['PrimaryKey'] == str(i + 1)]['Value'].to_list()))))
        time_tmp = []
        for j in range(len(df[df['PrimaryKey'] == str(i + 1)]['SourceTimeStamp'].to_list())):
            timestr = df[df['PrimaryKey'] == str(i + 1)]['SourceTimeStamp'].to_list()[j]
            datetime_obj = datetime.strptime(timestr, "%Y-%m-%dT%H:%M:%S.%fZ")
            obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
            time_tmp.append(obj_stamp)
        t.append(time_tmp)
    val = np.array(val)
    t = np.array(t)

    result1 = []
    for i in range(num_joints):
        result1.append([t[i], val[i]])

    val = []
    t = []
    for i in range(num_joints):
        val.append((list(map(float, df[df['PrimaryKey'] == str(i + 4)]['Value'].to_list()))))
        time_tmp = []
        for j in range(len(df[df['PrimaryKey'] == str(i + 4)]['SourceTimeStamp'].to_list())):
            timestr = df[df['PrimaryKey'] == str(i + 4)]['SourceTimeStamp'].to_list()[j]
            datetime_obj = datetime.strptime(timestr, "%Y-%m-%dT%H:%M:%S.%fZ")
            obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
            time_tmp.append(obj_stamp)
        t.append(time_tmp)
    val = np.array(val)
    t = np.array(t)

    result2 = []
    for i in range(num_joints):
        result2.append([t[i], val[i]])

    val = []
    t = []
    for i in range(num_joints):
        val.append((list(map(float, df[df['PrimaryKey'] == str(i + 7)]['Value'].to_list()))))
        time_tmp = []
        for j in range(len(df[df['PrimaryKey'] == str(i + 7)]['SourceTimeStamp'].to_list())):
            timestr = df[df['PrimaryKey'] == str(i + 7)]['SourceTimeStamp'].to_list()[j]
            datetime_obj = datetime.strptime(timestr, "%Y-%m-%dT%H:%M:%S.%fZ")
            obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
            time_tmp.append(obj_stamp)
        t.append(time_tmp)
    val = np.array(val)
    t = np.array(t)

    result3 = []
    for i in range(num_joints):
        result3.append([t[i], val[i]])

    return np.array(result1), np.array(result2), np.array(result3)

def load_csv_v2(file_name, num_joints):
    """
    Load structure with time data from csv file，有三个轴的actToolpos数值，还有三个编码器读数

    :param file_name: file path and file name
    :param num_joints: num of joints
    :returns: a numpy array shape:(num_joints, 2, time_n)；and a numpy array shape:(num_joints, 2, time_n)
    :raises keyError: raises an exception
    """
    col_list = ['PrimaryKey', 'DataType','Value','StatusCode','SourceTimeStamp','ServerTimeStamp']
    df = pd.read_csv(file_name, encoding='utf8', names=col_list)

    df = df.drop(index=[1,2,3])  # 删除第1、2、3行,1107增加

    val = []
    t = []
    for i in range(num_joints):
        val.append((list(map(float, df[df['PrimaryKey'] == str(i+1)]['Value'].to_list()))))
        time_tmp = []
        for j in range(len(df[df['PrimaryKey'] == str(i+1)]['SourceTimeStamp'].to_list())):
            timestr = df[df['PrimaryKey'] == str(i+1)]['SourceTimeStamp'].to_list()[j]
            datetime_obj = datetime.strptime(timestr, "%Y-%m-%dT%H:%M:%S.%fZ")
            obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
            time_tmp.append(obj_stamp)
        t.append(time_tmp)
    val = np.array(val)
    t = np.array(t)

    result1 = []
    for i in range(num_joints):
        result1.append([t[i],val[i]])

    val = []
    t = []
    for i in range(num_joints):
        val.append((list(map(float, df[df['PrimaryKey'] == str(i + 4)]['Value'].to_list()))))
        time_tmp = []
        for j in range(len(df[df['PrimaryKey'] == str(i + 4)]['SourceTimeStamp'].to_list())):
            timestr = df[df['PrimaryKey'] == str(i + 4)]['SourceTimeStamp'].to_list()[j]
            datetime_obj = datetime.strptime(timestr, "%Y-%m-%dT%H:%M:%S.%fZ")
            obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
            time_tmp.append(obj_stamp)
        t.append(time_tmp)
    val = np.array(val)
    t = np.array(t)

    result2 = []
    for i in range(num_joints):
        result2.append([t[i], val[i]])

    return np.array(result1), np.array(result2)

def load_csv(file_name, num_joints):
    """
    Load structure with time data from csv file，只有三个轴的actToolpos数值

    :param file_name: file path and file name
    :param num_joints: num of joints
    :returns: a numpy array shape:(num_joints, 2, time_n)
    :raises keyError: raises an exception
    """
    col_list = ['PrimaryKey', 'DataType','Value','StatusCode','SourceTimeStamp','ServerTimeStamp']
    df = pd.read_csv(file_name, encoding='utf8', names=col_list)
    val = []
    t = []
    for i in range(num_joints):
        val.append((list(map(float, df[df['PrimaryKey'] == str(i+1)]['Value'].to_list()))))
        time_tmp = []
        for j in range(len(df[df['PrimaryKey'] == str(i+1)]['SourceTimeStamp'].to_list())):
            timestr = df[df['PrimaryKey'] == str(i+1)]['SourceTimeStamp'].to_list()[j]
            datetime_obj = datetime.strptime(timestr, "%Y-%m-%dT%H:%M:%S.%fZ")
            obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
            time_tmp.append(obj_stamp)
        t.append(time_tmp)
    val = np.array(val)
    t = np.array(t)

    result = []
    for i in range(num_joints):
        result.append([t[i],val[i]])
    return np.array(result)

def interp_list(iterm):
    """
    Interpolate sequence iterm.

    :param iterm: a numpy array shape:(num_joints, 2, time_n)
    :returns: a numpy array shape:(2(xy), t(n*1ms))
    :raises keyError: raises an exception
    """
    r = max(max(iterm[0][0]),max(iterm[1][0]))
    l = min(min(iterm[0][0]), min(iterm[1][0]))
    result = []
    result_t = []
    for i in range(2):
        x = iterm[i][0]
        y = iterm[i][1]
        #插入点为xvals
        xvals = np.linspace(l,r,r-l)
        yinterp = np.interp(xvals, x, y)
        result.append(yinterp)
        result_t.append([xvals,yinterp])
    return np.array(result),np.array(result_t)

def interp_list_only_sample_points(iterm):
    """
    Interpolate sequence iterm.

    :param iterm: a numpy array shape:(num_joints, 2, time_n)
    :returns: a numpy array shape:(2(xy),2, t(sample_points*1ms))
    :raises keyError: raises an exception
    """

    result = []
    for i in range(2):
        x = iterm[i][0]
        y = iterm[i][1]
        # 插入点为xvals
        xvals = np.union1d(iterm[0][0], iterm[1][0])
        # print(xvals)
        yinterp = np.interp(xvals, x, y)
        result.append([xvals, yinterp])
    return np.array(result)

def interp_list_only_sample_points_yz(iterm):
    """
    Interpolate sequence iterm.

    :param iterm: a numpy array shape:(num_joints, 2, time_n)
    :returns: a numpy array shape:(2(xy),2, t(sample_points*1ms))
    :raises keyError: raises an exception
    """

    result = []
    for i in range(2):
        x = iterm[i+1][0]
        y = iterm[i+1][1]
        # 插入点为xvals
        xvals = np.union1d(iterm[1][0], iterm[2][0])
        # print(xvals)
        yinterp = np.interp(xvals, x, y)
        result.append([xvals, yinterp])
    return np.array(result)

def get_list_only_sample_point_act(file_path, file_name, offset):
    """
    NO Interpolate sequence iterm acttoolbasepos.

    :returns: a numpy array shape:(2(xy), 2, t(sample_points*1ms))
    :raises keyError: raises an exception
    """

    x = offset[0]
    y = offset[1]
    z = 0

    col_list = ['PrimaryKey', 'DataType', 'Value', 'StatusCode', 'SourceTimeStamp', 'ServerTimeStamp']
    df = pd.read_csv(file_path+file_name+'.csv', encoding='utf8', names=col_list)
    data = df[['PrimaryKey', 'Value', 'SourceTimeStamp']].values
    data = data[1:]
    for i in range(len(data)):
        timestr = data[i][2]
        datetime_obj = datetime.strptime(timestr, "%Y-%m-%dT%H:%M:%S.%fZ")
        obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
        data[i][2] = obj_stamp
    data = np.array(data, dtype=float)

    result = []
    for i in range(data.shape[0]):
        if data[i][0] == 1:
            for j in range(len(result)):
                if result[len(result)-1-j][0][0] == data[i][2]:
                    result[len(result)-1-j][0][1] = data[i][1]
                    break
            else:
                x = data[i][1]
                result.append(np.array([[data[i][2],x],[data[i][2],y]]))
                continue
            #break
        elif data[i][0] == 2:
            for j in range(len(result)):
                if result[len(result) - 1 - j][0][0] == data[i][2]:
                    result[len(result) - 1 - j][1][1] = data[i][1]
                    break
            else:
                y = data[i][1]
                result.append(np.array([[data[i][2],x],[data[i][2],y]]))
                continue
            #break
    result = np.array(result)
    result = result.transpose(1,2,0)
    return result

def get_list_only_sample_point_encoder(file_path, file_name, offset):
    """
    NO Interpolate sequence iterm encoder pos.

    :returns: a numpy array shape:(2(xy), 2, t(sample_points*1ms))
    :raises keyError: raises an exception
    """

    x = offset[0]
    y = offset[1]
    z = 0

    col_list = ['PrimaryKey', 'DataType', 'Value', 'StatusCode', 'SourceTimeStamp', 'ServerTimeStamp']
    df = pd.read_csv(file_path+file_name+'.csv', encoding='utf8', names=col_list)
    data = df[['PrimaryKey', 'Value', 'SourceTimeStamp']].values
    data = data[1:]
    for i in range(len(data)):
        timestr = data[i][2]
        datetime_obj = datetime.strptime(timestr, "%Y-%m-%dT%H:%M:%S.%fZ")
        obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
        data[i][2] = obj_stamp
    data = np.array(data, dtype=float)

    result = []
    for i in range(data.shape[0]):
        if data[i][0] == 4:
            for j in range(len(result)):
                if result[len(result)-1-j][0][0] == data[i][2]:
                    result[len(result)-1-j][0][1] = data[i][1]
                    break
            else:
                x = data[i][1]
                result.append(np.array([[data[i][2],x],[data[i][2],y]]))
                continue
            #break
        elif data[i][0] == 5:
            for j in range(len(result)):
                if result[len(result) - 1 - j][1][0] == data[i][2]:#这里有问题吗
                    result[len(result) - 1 - j][1][1] = data[i][1]
                    break
            else:
                y = data[i][1]
                result.append(np.array([[data[i][2],x],[data[i][2],y]]))
                continue
            #break
    result = np.array(result)
    result = result.transpose(1,2,0)
    return result

def get_list_only_sample_point_vel(file_path, file_name, offset):
    """
    NO Interpolate sequence iterm encoder pos.

    :returns: a numpy array shape:(2(xy), 2, t(sample_points*1ms))
    :raises keyError: raises an exception
    """

    x = offset[0]
    y = offset[1]
    z = 0

    col_list = ['PrimaryKey', 'DataType', 'Value', 'StatusCode', 'SourceTimeStamp', 'ServerTimeStamp']
    df = pd.read_csv(file_path+file_name+'.csv', encoding='utf8', names=col_list)
    data = df[['PrimaryKey', 'Value', 'SourceTimeStamp']].values
    data = data[1:]
    for i in range(len(data)):
        timestr = data[i][2]
        datetime_obj = datetime.strptime(timestr, "%Y-%m-%dT%H:%M:%S.%fZ")
        obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
        data[i][2] = obj_stamp
    data = np.array(data, dtype=float)

    result = []
    for i in range(data.shape[0]):
        if data[i][0] == 7:
            for j in range(len(result)):
                if result[len(result)-1-j][0][0] == data[i][2]:
                    result[len(result)-1-j][0][1] = data[i][1]
                    break
            else:
                x = data[i][1]
                result.append(np.array([[data[i][2],x],[data[i][2],y]]))
                continue
            #break
        elif data[i][0] == 8:
            for j in range(len(result)):
                if result[len(result) - 1 - j][1][0] == data[i][2]:#这里有问题吗
                    result[len(result) - 1 - j][1][1] = data[i][1]
                    break
            else:
                y = data[i][1]
                result.append(np.array([[data[i][2],x],[data[i][2],y]]))
                continue
            #break
    result = np.array(result)
    result = result.transpose(1,2,0)
    return result

def g54_operation(pos_xy,offset):
    """
    Convert wcs to mcs

    :param pos_xy: a numpy array shape:(2(x,y), time_n)
    :param offset: offset (x,y)
    :returns: a numpy array shape:(2(xy), t(n*1ms))
    :raises keyError: raises an exception
    """
    for i in range(pos_xy.shape[0]):
        pos_xy[i][0] = pos_xy[i][0] + offset[0]
        pos_xy[i][1] = pos_xy[i][1] + offset[1]
    return pos_xy

def inverse_g54_operation(pos_xy,offset):
    """
    Convert mcs to wcs

    :param pos_xy: a numpy array shape:(2(x,y), time_n)
    :param offset: offset (x,y)
    :returns: a numpy array shape:(2(xy), t(n*1ms))
    :raises keyError: raises an exception
    """
    for i in range(pos_xy.shape[1]):
        pos_xy[0][i] = pos_xy[0][i] - offset[0]
        pos_xy[1][i] = pos_xy[1][i] - offset[1]
    return pos_xy

def plot_track(file_path, file_name, offset):
    '''
    画一个曲线
    :param file_path:
    :param file_name:
    :param offset:
    :return:
    '''
    data,data_encoder = load_csv_v2(file_path+file_name + '.csv', 3)
    print("data.shape: ", data.shape)


    print("data_encoder.shape: ", data_encoder.shape)

    #h_only_sample_points = get_list_only_sample_point_encoder(file_path, file_name, offset)
    h_only_sample_points = interp_list_only_sample_points(data_encoder)

    start_proportion = 0
    end_proportion = 1
    # plt = plot_current_data.create_figure((35,6),'time','current')
    plt = plot_current_data.create_figure((50, 6), 'time', 'opcua')
    # plt.title(file_name)

    plt = plot_current_data.scatter_series_with_time(plt, h_only_sample_points[0][:, int(start_proportion * h_only_sample_points.shape[2]):int(
        end_proportion * h_only_sample_points.shape[2])], "x")
    plt = plot_current_data.scatter_series_with_time(plt, h_only_sample_points[1][:, int(start_proportion * h_only_sample_points.shape[2]):int(
        end_proportion * h_only_sample_points.shape[2])], "y")

    plt.show()

    h_only_sample_points = h_only_sample_points[:,1,:]

    print("h_only_sample_points.shape: ",h_only_sample_points.shape)

    h_only_sample_points = inverse_g54_operation(np.array(h_only_sample_points), offset)  # 从机床坐标系转变为工件坐标系
    plt.figure(figsize=(10,10))
    plt.scatter(np.array(h_only_sample_points)[0], np.array(h_only_sample_points)[1])
    plt.show()

if __name__ == '__main__':
    # plot_track('C:\\Users\lsj\PycharmProjects\experiment0616\\poly\opcua\\','DataLogger_20231012-XYshift1_F500_BRISK-0001',[-148.55306, -295.10247])

    # file_path = 'C:\\Users\lsj\PycharmProjects\experiment0616\\'
    file_path = 'C:\\Users\lsj\PycharmProjects\840D\data0119\\'
    # file_name = 'DataLogger_20230804-fangyuanjian-F500-50hz-0001-1691135544517-1664'
    # file_name = 'DataLogger_20230804-zimu-F500-50hz-0001-1691148864739-1976'
    # file_name = 'DataLogger_hole_10_3'
    # file_name = 'DataLogger_hole_normal'
    file_name = 'DataLogger_hole_20_6'


    offset = [-200.31935, -225.37190]

    data, data_encoder = load_csv_v2(file_path + file_name + '.csv', 3)
    print("data.shape: ", data.shape)
    print("data_encoder.shape: ", data_encoder.shape)
    h_only_sample_points = interp_list_only_sample_points(data_encoder)
    print("(h_only_sample_points.shape: ",h_only_sample_points.shape)

    h_only_sample_points[0][1] = h_only_sample_points[0][1] - offset[0]
    h_only_sample_points[1][1] = h_only_sample_points[1][1] - offset[1]
    #
    # h_only_sample_points_act = interp_list_only_sample_points(data)

    h, h_t = interp_list(data)
    print("h_t.shape: ", h_t.shape)
    print("h.shape: ", h.shape)
    # np.save(file_path + 'movement/fangyuanjian_pos_opcua.npy',h_t,allow_pickle=True)
    # np.save(file_path + 'movement/zimu_pos_opcua.npy', h_t, allow_pickle=True)
    # reverse_point = find_poly_start_point(h_t, offset)
    # print("len(reverse_point): ", len(reverse_point))


    x_ini, y_ini, z_ini = 0, 0, 0
    instruction = [['straight', '5', '0', '0'], ['straight', '5', '5', '0'],['arc','5','5','0','5','10.5',-1], ['straight', '5', '0', '0'],['straight', '10', '0', '0']]  #


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

    # 所有点ketree
    dist_h_only_sample_points = []
    for i in range(h_only_sample_points.shape[2]):
        target_point = h_only_sample_points[:, 1, i]
        # 在加载后的KD-tree中查找距离目标点最近的点
        distance, index = kdtree.query(target_point)
        # nearest_point = point_collection[index]
        dist_h_only_sample_points.append(distance)

    dist_h_only_sample_points = np.array(
        [h_only_sample_points[0][0], dist_h_only_sample_points])
    print("dist_h_only_sample_points.shape: ", dist_h_only_sample_points.shape)
    print(np.mean(dist_h_only_sample_points[1]))

    plt.figure()
    plt.scatter(point_collection[:,0], point_collection[:,1])
    plt.scatter(h_only_sample_points[0][1], h_only_sample_points[1][1])
    plt.show()


    plt.figure()
    plt.scatter(dist_h_only_sample_points[0],dist_h_only_sample_points[1])
    plt.show()