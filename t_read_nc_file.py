
import re

import numpy as np

def read_nc_file( file_name ):

    with open( file_name, 'r' ) as f_id:

        lines = f_id.readlines()

    return lines
def extract_data_this_line( this_line= None  ):

    text = [ 'X' , 'Y' , 'Z' ]

    # this_line = ' X4.777 Y1.478\n'

    this_line_2 = this_line.replace( '\n' , '')

    #print( this_line_2 )

    m_x = re.finditer( 'X' , this_line_2 )
    m_y = re.finditer( 'Y' , this_line_2 )




    for x in m_x:
        idx_x = x.start()
        #print( 'idx_y' , idx_x )
        break

    for y in m_y:
        idx_y = y.start()
        #print('idx_x'  , idx_y)
        break



    value_str_x = this_line_2[ idx_x+1:idx_y]
    #print( value_str_x )

    value_str_y = this_line_2[idx_y+1:]
    #print(value_str_y)

    value_x = float( value_str_x)
    value_y = float( value_str_y )

    pos_xy = np.array( [value_x, value_y])

    # print( pos_xy)

    return  pos_xy

def extract_pos_all_lines( lines ):

    pos_xy = np.empty((0,2))
    for this_line in lines:
        try:
            pos_this_line = extract_data_this_line(this_line=this_line)

            pos_xy = np.r_[ pos_xy, np.array([ pos_this_line]) ]
        except Exception as e:
            print(e )

    # pos_xy = pos_xy.transpose()


    return pos_xy

# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     file_name = r'.\_N_POLY_AT0602_SPF_shiyitu'
#
#     lines = read_nc_file( file_name )
#
#     print( lines )
#
#     # extract_data_this_line()
#     pos_xy = extract_pos_all_lines( lines)
#     print("pos_xy.shape: ",pos_xy.shape)
#     plt.figure()
#     plt.plot( pos_xy)
#     plt.figure()
#     plt.plot( pos_xy[:,0] , pos_xy[:,1])
#     plt.show()