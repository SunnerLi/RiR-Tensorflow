from matplotlib import pyplot as plt
import numpy as np

def draw(loss_dict):
    """
        Draw the line plot with points
        (You should call meanError to get the mean list before calling this function)

        Arg:    loss_dict   - The dict object which contain training mean result
    """
    plt.figure(1)

    # Plot loss
    plt.subplot(211)
    curve_line_1, = plt.plot(loss_dict['loss']['cnn'], 'k', color='g', label='CNN')
    curve_point_1, = plt.plot(loss_dict['loss']['cnn'], 'bo', color='g')
    curve_line_2, = plt.plot(loss_dict['loss']['resnet'], 'k', color='b', label='ResNet')
    curve_point_2, = plt.plot(loss_dict['loss']['resnet'], 'bo', color='b')
    curve_line_3, = plt.plot(loss_dict['loss']['rir'], 'k', color='orange', label='RiR')
    curve_point_3, = plt.plot(loss_dict['loss']['rir'], 'bo', color='orange')
    plt.legend(handles=[curve_line_1, curve_line_2, curve_line_3])
    plt.title('Loss')

    # Plot accuracy
    plt.subplot(212)
    curve_line_1, = plt.plot(loss_dict['acc']['cnn'], 'k', color='g', label='CNN')
    curve_point_1, = plt.plot(loss_dict['acc']['cnn'], 'bo', color='g')
    curve_line_2, = plt.plot(loss_dict['acc']['resnet'], 'k', color='b', label='ResNet')
    curve_point_2, = plt.plot(loss_dict['acc']['resnet'], 'bo', color='b')
    curve_line_3, = plt.plot(loss_dict['acc']['rir'], 'k', color='orange', label='RiR')
    curve_point_3, = plt.plot(loss_dict['acc']['rir'], 'bo', color='orange')
    plt.legend(handles=[curve_line_1, curve_line_2, curve_line_3])
    plt.title('Accuracy')

    # Show
    plt.savefig('record.png')                                                 # save before show to avoid refreshing
    plt.show()
    

def meanError(loss_dict, scalar):
    for record_type in loss_dict.keys():
        for net_type in loss_dict[record_type].keys():
            origin_list = np.reshape(loss_dict[record_type][net_type], [scalar, -1])
            loss_dict[record_type][net_type] = np.asarray(np.mean(origin_list, axis=0))
    return loss_dict

if __name__ == '__main__':
    loss_dict = {
        'cnn': [2.1, 2.0, 1.9, 2.1, 2.0, 1.9],
        'resnet': [2.1, 1.9, 1.8, 2.1, 2.0, 1.9],
        'rir': [2.05, 1.85, 1.7, 2.1, 2.0, 1.9]
    }
    loss_dict = meanError(loss_dict, 2)
    draw(loss_dict)