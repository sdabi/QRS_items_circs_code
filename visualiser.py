import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import defines
import matplotlib.cm as cm

from math import atan2, degrees

# Label line with line2D label data
def labelLine(line, x, label=None, align=True, **kwargs):
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    # Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip - 1] + (ydata[ip] - ydata[ip - 1]) * (x - xdata[ip - 1]) / (xdata[ip] - xdata[ip - 1])

    if not label:
        label = line.get_label()

    if align:
        # Compute the slope
        dx = xdata[ip] - xdata[ip - 1]
        dy = ydata[ip] - ydata[ip - 1]
        ang = degrees(atan2(dy, dx))

        # Transform to screen co-ordinates
        pt = np.array([x, y]).reshape((1, 2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)), pt)[0]

    else:
        trans_angle = 0

    # Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x, y, label, rotation=trans_angle, **kwargs)


def labelLines(lines, align=True, xvals=None, **kwargs):
    ax = lines[0].axes
    labLines = []
    labels = []

    # Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin, xmax = ax.get_xlim()
        xvals = np.linspace(xmin, xmax, len(labLines) + 2)[1:-1]

    for line, x, label in zip(labLines, xvals, labels):
        labelLine(line, x, label, align, **kwargs)


RED = (200, 50, 50)
GREEN = (0, 200, 100)
BLUE = (35, 100, 255)
GRAY = (150, 150, 150)
ORANGE = (255, 165, 30)
BLACK = (0, 0, 0)

def bold(text):
    return "\033[1m{}\033[0m".format(text)

def underline(text):
    return "\033[4m{}\033[0m".format(text)

def colored(color, text):
    return "\033[38;2;{};{};{}m{}\033[0m".format(color[0], color[1], color[2], bold(text))





def plot_embedded_vecs(vecs):
    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(vecs)

    max_diff_x = max(vecs_2d,key=lambda item:item[0])[0] - min(vecs_2d,key=lambda item:item[0])[0]
    max_diff_y = max(vecs_2d,key=lambda item:item[1])[1] - min(vecs_2d,key=lambda item:item[1])[1]


    for i, point in enumerate(vecs_2d):
        x, y = point[0], point[1]
        plt.scatter(x, y, color=cm.tab20b(i))
        plt.annotate(i, (x, y),
                     xytext=(x+(max_diff_x/32), y+(max_diff_y/32)), fontsize=8, color=cm.tab20b(i%20), weight='bold')
    plt.show()

def plot_embedded_vecs_3d(vecs):
    pca = PCA(n_components=3)
    vecs_3d = pca.fit_transform(vecs)

    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    for i, xyz_ in enumerate(vecs_3d):
        x = xyz_[0]
        y = xyz_[1]
        z = xyz_[2]
        label = i
        ax_3d.scatter(x, y, z, color='b')
        ax_3d.text(x, y, z, '%s' % (label), size=20, zorder=1, color='k')
    plt.show()




def plot_HRK(hrk_list):
    y_axis = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()

    for HRK, title in hrk_list:
        ax.plot(HRK, label=title)

    plt.ylim([0, 1.1])
    plt.yticks(y_axis)
    plt.xlim([1, 11])
    labelLines(plt.gca().get_lines(),zorder=2.5)
    plt.grid()
    plt.show()

def print_colored_matrix(mat, colored_values=[], is_vec=0, all_positive=0, digits_after_point=1, print_col_index=0):
    space_for_print = (1-all_positive)+digits_after_point+3

    if print_col_index:
        print("     ", end="")
        for i in range(defines._NUM_OF_ITEMS):
            print(colored(GRAY, underline('{val:>{space_for_print}}'.format(val=i, space_for_print=space_for_print))), end="")

    r = -1
    # print("")  # new line
    # print(colored(GRAY, '{val:>{space_for_print}}'.format(val=r, space_for_print=3)), colored(GRAY, "|"), end="")

    # code which mark the higest prop in the vec
    if is_vec == 1:
        colored_values.append([])
        colored_values[3].append(np.argmax(np.array(mat)))

    for index_in_mat, val in np.ndenumerate(mat):
        print_in_black = 1
        if is_vec == 0 and r < index_in_mat[0]:
            print("")  # new line
            r = index_in_mat[0]
            print(colored(GRAY, '{val:>{space_for_print}}'.format(val=r, space_for_print=3)), colored(GRAY, "|"), end="")

        val = '{val:1.{digits_after_point}f}'.format(val=val, digits_after_point=digits_after_point)
        for index, color_list in enumerate(colored_values):
            # print("A", index_in_mat, index, color_list)
            if index_in_mat in color_list:
                # print("B", index_in_mat, index, color_list)
                print_in_black = 0
                if index == 0:
                    print(colored(RED, '{val:>{space_for_print}}'.format(val=val, space_for_print=space_for_print)), end="")
                    break
                if index == 1:
                    print(colored(GREEN, '{val:>{space_for_print}}'.format(val=val, space_for_print=space_for_print)), end="")
                    break
                if index == 2:
                    print(colored(BLUE, '{val:>{space_for_print}}'.format(val=val, space_for_print=space_for_print)), end="")
                    break
                if index == 3:
                    print(colored(ORANGE, '{val:>{space_for_print}}'.format(val=val, space_for_print=space_for_print)), end="")
                    break
        if print_in_black:
            print( '{val:>{space_for_print}}'.format(val=val, space_for_print=space_for_print), end="")

    print("")
    return

def plot_cost_arrs(cost_arrs):
    cmap = plt.get_cmap('tab20')
    fig = plt.figure(figsize=(10, 20))
    ax = fig.add_subplot()

    x = range(len(cost_arrs[0]))
    for i, cost_arr in enumerate(cost_arrs):
        ax.plot(x, cost_arr, color=cmap(i), label=i, linewidth=2)
    ax.legend(range(len(cost_arrs)), loc='upper right')

    ymax = max(max(l) for l in cost_arrs)
    plt.ylim(ymin=0.0001,ymax=ymax)
    plt.ylabel('cost')
    plt.xlabel('iteration')

    plt.xlim(xmin=0,xmax=len(x)-1)
    plt.show()



def print_reco_matrix(mat, removed_inter_indicies, values=[], all_positive=1, digits_after_point=0):
    space_for_print = (1 - all_positive) + digits_after_point + 3

    if len(values) == 0:
        mat_to_print = mat
    else:
        mat_to_print = values

    print("     ", end="")
    for i in range(defines._NUM_OF_ITEMS):
        print(colored(GRAY, underline('{val:>{space_for_print}}'.format(val=i, space_for_print=space_for_print))), end="")
    print("") # new line
    for row_num, vec in enumerate(mat):
        print(colored(GRAY, '{val:>{space_for_print}}'.format(val=row_num, space_for_print=3)), colored(GRAY, "|"), end="")
        for col_num, val in enumerate(vec):
            val_to_print = mat_to_print[row_num][col_num]
            if digits_after_point:
                val_to_print = '{val:1.{digits_after_point}f}'.format(val=val_to_print, digits_after_point=digits_after_point)
            if (row_num, col_num) in removed_inter_indicies:
                print(colored(BLUE, '{val:>{space_for_print}}'.format(val=val_to_print, space_for_print=space_for_print)), end="")
                continue
            if val == 0:
                print('{val:>{space_for_print}}'.format(val=val_to_print, space_for_print=space_for_print),end="")
            if val == 1:
                print(colored(GREEN, '{val:>{space_for_print}}'.format(val=val_to_print, space_for_print=space_for_print)), end="")
            if val == -1:
                print(colored(RED, '{val:>{space_for_print}}'.format(val=val_to_print, space_for_print=space_for_print)), end="")
        print("") # new line
    return


