import re
import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import sys  # 导入sys模块
sys.setrecursionlimit(5000)  # 将默认的递归深度修改为3000
import pickle
import h5py

plane = 1#1为XY平面；2为YZ平面；3为XZ平面


file_path = './gcode/'
# file_name = 'O3333_v3_singlelayer_mm'#

def Gcode_parser(file_path, file_name):
    f = open(file_path + file_name + '.txt')
    data = f.readlines()
    f.close()

    instruction = []

    for i in data:
        pattern1 = r"STRAIGHT_TRAVERSE\((-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\)"
        pattern2 = r"STRAIGHT_FEED\((-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\)"
        pattern3 = r"ARC_FEED\((-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\)"

        matches1 = re.findall(pattern1, i.strip())
        matches2 = re.findall(pattern2, i.strip())
        matches3 = re.findall(pattern3, i.strip())

        if matches1:
            x, y, z, a, b, c = matches1[0]
            instruction.append(['straight',x, y, z, a, b, c])
            # print(x, y, z, a, b, c)
            continue

        if matches2:
            x, y, z, a, b, c = matches2[0]
            instruction.append(['straight', x, y, z, a, b, c])
            # print(x, y, z, a, b, c)
            continue

        if matches3:
            e1, e2, c1, c2, rotation, e_other, a, b, c = matches3[0]
            if plane == 1:
                instruction.append(['arc', e1, e2, e_other, c1, c2, rotation])
            elif plane == 2:
                instruction.append(['arc', e_other, e1, e2, c1, c2, rotation])
            elif plane == 3:
                instruction.append(['arc', e2, e_other, e1, c1, c2, rotation])
            print(e1, e2, c1, c2, rotation, e_other, a, b, c)
            print(i)
            continue
    return instruction

# instruction = np.array(instruction)
# print(instruction.shape)

# point = instruction[:,1:4].astype(float)
# print(point.shape)

def calculate_angle(x, y):
    angle_rad = np.arctan2(y, x)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def line_interpolate(x1, y1, z1, x2, y2, z2):
    num_points = int((((x1 - x2)**2 + (y1 - y2)**2)**0.5)/0.003)
    x_values = np.linspace(x1, x2, num_points)
    y_values = np.linspace(y1, y2, num_points)
    interpolated_points = np.column_stack((x_values, y_values))
    return interpolated_points

def line_interpolate_5axis(x1, y1, z1, b1, c1, x2, y2, z2, b2, c2):
    num_points = int((((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**0.5)/0.003)
    # print(num_points)
    x_values = np.linspace(x1, x2, num_points)
    y_values = np.linspace(y1, y2, num_points)
    z_values = np.linspace(z1, z2, num_points)
    b_values = np.linspace(b1, b2, num_points)
    c_values = np.linspace(c1, c2, num_points)
    # interpolated_points = np.column_stack((x_values, y_values))
    interpolated_points = np.array([x_values, y_values,z_values,b_values,c_values])
    interpolated_points = interpolated_points.transpose(1, 0)
    return interpolated_points

def arc_interpolation(x1, y1, z1, x2, y2, z2, c1, c2, plane, rotation):
    '''
    :param x1: 起点坐标
    :param y1: 起点坐标
    :param z1: 起点坐标
    :param x2: 终点坐标
    :param y2: 终点坐标
    :param z2: 终点坐标
    :param c1: 圆心坐标
    :param c2: 圆心坐标
    :param plane: 平面选择
    :param rotation: 顺逆时针
    :return:
    '''
    if plane == 1:

        # print("起点坐标：",x1,y1)

        r = ((x1 - c1)**2 + (y1 - c2)**2)**0.5
        # print("半径: ",r)

        # 圆心坐标和半径
        cx, cy = c1, c2
        # print("圆心坐标: ",cx,cy)

        # 角度范围
        start_angle = calculate_angle(x1-c1, y1-c2)  # 起始角度（单位：度）
        end_angle = calculate_angle(x2-c1, y2-c2)  # 终止角度（单位：度）

        # if(x1 == x2) & (y1 == y2):
        #     start_angle = 0
        #     end_angle = 360

        if rotation == 1:
            if end_angle <= start_angle:
                end_angle = end_angle + 360
        elif rotation == -1:
            if end_angle >= start_angle:
                end_angle = end_angle - 360

        # print("起始角度: ",start_angle)
        # print("终止角度: ",end_angle)

        # 将角度转换为弧度
        start_angle_rad = np.radians(start_angle)
        end_angle_rad = np.radians(end_angle)

        if rotation == 1:
            step = 360 * 0.003/(2*np.pi*r)
        elif rotation == -1:
            step = -1 * 360 * 0.003/(2*np.pi*r)
            # step = -1

        arc_points = []

        for angle_rad in np.arange(start_angle_rad, end_angle_rad, np.radians(step)):
            x = cx + r * np.cos(angle_rad)
            y = cy + r * np.sin(angle_rad)
            arc_points.append((x, y))

        arc_points = np.array(arc_points)
        return arc_points
        print(arc_points.shape)

if __name__ == '__main__':
    # file_name = 're_041'  #
    # instruction = Gcode_parser(file_path,file_name)
    # print(len(instruction))
    # # # x_ini, y_ini, z_ini = 0,0,0
    # # # instruction = [['straight','10','10','0'],['straight','0','20','0'],['straight','-10','10','0'],['straight','0','0','0'],['arc','10','0','0','5','0',-1]]#
    # x_ini, y_ini, z_ini, b_ini, c_ini = 0,0,0,0,0
    #
    # point_collection = np.array([[0,0,0,0,0]])
    # print("point_collection.shape: ",point_collection.shape)
    #
    # nums = 0
    #
    # for i in instruction:
    #     print(nums)
    #     nums = nums + 1
    #     if i[0] == 'straight':
    #         # print(i)
    #         # print(float(i[1]), float(i[2]), float(i[3]), float(i[4]), float(i[5]))
    #         # line_points = line_interpolate(x_ini, y_ini, z_ini, float(i[1]), float(i[2]), float(i[3]))
    #         line_points = line_interpolate_5axis(x_ini, y_ini, z_ini, b_ini, c_ini, float(i[1]), float(i[2]), float(i[3]), float(i[5]), float(i[6]))
    #         x_ini, y_ini, z_ini, b_ini, c_ini = float(i[1]), float(i[2]), float(i[3]), float(i[5]), float(i[6])
    #         if len(line_points) == 0:
    #             continue
    #         # print("line_points.shape: ",line_points.shape)
    #         point_collection = np.concatenate((point_collection,line_points),axis=0)
    #     elif i[0] == 'arc':
    #         print('arc')
    #     # print(i)
    #     print(point_collection.shape)
    #
    # print("point_collection.shape: ",point_collection.shape)
    #
    # np.save('./point_collection',point_collection)

    point_collection = np.load('./point_collection.npy')
    print("point_collection.shape: ", point_collection.shape)

    x = point_collection[:,0]
    y = point_collection[:,1]
    z = point_collection[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    #
    # target_point = [1,1]
    #
    # kdtree = KDTree(point_collection)
    #
    # distance, index = kdtree.query(target_point)
    #
    # nearest_point = point_collection[index]
    #
    #
    # plt.figure(figsize=(10,10))
    # # plt.scatter(arc_points[:,0],arc_points[:,1])
    # plt.scatter(point_collection[:,0],point_collection[:,1])
    # plt.show()
    #
    # # mpl.rcParams['legend.fontsize'] = 10
    # #
    # # fig = plt.figure(figsize=(10, 10))
    # # ax = fig.gca(projection='3d')
    # # ax.set_xlabel('X/mm', size=15, labelpad=15)
    # # ax.set_ylabel('Y/mm', size=15, labelpad=15)
    # # ax.set_zlabel('Z/mm', size=15, labelpad=15)
    # #
    # # ax.tick_params(labelsize=15)
    # #
    # # # ax.set_xlim([-0.1, 1.1])
    # # # ax.set_ylim([-0.1, 1.1])
    # # # ax.set_zlim([-0.1, 1.1])
    # #
    # # # ax.set_xticks([0.0, 1.0])
    # # # ax.set_zticks([0.0, 1.0])
    # # # ax.set_yticks([0.0, 1.0])
    # #
    # # ax.scatter(point_collection[:,0], point_collection[:,1], point_collection[:,2], label='point_collection')
    # # ax.legend()
    # #
    # # plt.show()
    #
    # h5f = h5py.File(file_path+file_name+'.h5', 'w')
    # h5f.create_dataset('point_collection', data=point_collection)
    # h5f.close()


    #
    # print(type(kdtree))
    #
    # with open(file_path+file_name+'.pkl', 'wb') as f:
    #     pickle.dump(kdtree, f)

    # line_points = line_interpolate(5,0,0,0,5,0)
    # arc_points = arc_interpolation(0,5,0,0,-5,0,0,0,1,0)
    #
    # plt.figure(figsize=(10,10))
    # # plt.scatter(arc_points[:,0],arc_points[:,1])
    # plt.scatter(line_points[:,0],line_points[:,1])
    # plt.show()
