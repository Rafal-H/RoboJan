from configparser import ConfigParser
import numpy as np

def get_parameters(filepath):
    config = ConfigParser()
    config.read(filepath)
    setup = dict(config['SETUP'])
    for key in setup:
        setup[key] = setup[key].split(', ')
        for i in range(len(setup[key])):
            setup[key][i] = int(setup[key][i])

    if config.has_section('CALIBRATION'):
        calibration = dict(config['CALIBRATION'])
        for key in calibration:
            calibration[key] = calibration[key].split()
        return setup, calibration
    else:
        calibration = False
        return setup, calibration


def generate_object_points(setup):
    objectpoints = np.array([0, 0, 0, setup['runway_length'][0], 0, 0, setup['runway_length'][0], setup['runway_width'][0], 0, 0, setup['runway_width'][0], 0], np.float32).reshape(4,3)
    return objectpoints
