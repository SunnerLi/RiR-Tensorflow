from matplotlib import pyplot as plt
import numpy as np

def draw(loss_dict):
    curve_line_1, = plt.plot(loss_dict['cnn'], 'k', color='g', label='CNN')
    curve_point_1, = plt.plot(loss_dict['cnn'], 'bo', color='g')
    curve_line_2, = plt.plot(loss_dict['resnet'], 'k', color='b', label='ResNet')
    curve_point_2, = plt.plot(loss_dict['resnet'], 'bo', color='b')
    curve_line_3, = plt.plot(loss_dict['rir'], 'k', color='orange', label='RiR')
    curve_point_3, = plt.plot(loss_dict['rir'], 'bo', color='orange')
    plt.legend(handles=[curve_line_1, curve_line_2, curve_line_3])
    plt.show()
    plt.savefig('loss.png')

def meanError(loss_dict, scalar):
    for key in loss_dict.keys():
        origin_list = np.reshape(loss_dict[key], [scalar, -1])
        loss_dict[key] = np.asarray(np.mean(origin_list, axis=0))
        # print(loss_dict[key])
    return loss_dict

if __name__ == '__main__':
    loss_dict = {
        'cnn': [2.1, 2.0, 1.9, 2.1, 2.0, 1.9],
        'resnet': [2.1, 1.9, 1.8, 2.1, 2.0, 1.9],
        'rir': [2.05, 1.85, 1.7, 2.1, 2.0, 1.9]
    }
    loss_dict = meanError(loss_dict, 2)
    draw(loss_dict)