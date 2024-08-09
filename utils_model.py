import sys


def get_network(net_type,num_classes=100):

    if net_type == 'inceptionv3':
        from model_template.inceptionv3 import inceptionv3
        net = inceptionv3(num_classes=num_classes)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net
