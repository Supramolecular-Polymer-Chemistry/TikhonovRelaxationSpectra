import re

# -----------------------------------------------------------------------------
#
# Python script for rewriting TRIOS data to files expected by the
# Tikhonov relaxation spectra script
#
#   Version 1.0
# Author:   Roy Wink, TU Eindhoven, 2026
# License:  GNU GPLv3
#
# --- SETTINGS ----------------------------------------------------------------

filename = 'data//trios_stressrelaxation.txt'

cutoff_time = 0.1
normalize = False

# -----------------------------------------------------------------------------

def get_stressrelaxation(filename):
    # open the specified file
    with open(filename, 'r') as f:
        data = f.read()

    # replace all commas for periods
    data = data.replace(',', '.')

    # split data at [step] flags
    data = data.split('[step]\n')

    # retreive stress relaxations
    stress_relaxations = []
    for step in data[1:]:
        if re.match('^(Stress relaxation)', step):
            stress_relaxations.append(step)

    # check amount of stress relaxations
    if len(stress_relaxations) == 0:
        raise AssertionError(f'No stress relaxations in {filename}')

    return stress_relaxations


def normalise_array(array):
    return [x / array[0] for x in array]


def parse_stressrelaxation(step, cutoff_time, step_num):
    # split and remove empty lines
    step_data = step.split('\n')[1:]
    step_data[:] = [i for i in step_data if i]

    # get parameter and unit data
    parameters = step_data[0].split('\t')
    # units = sr_data[1].split('\t')

    # get temperature of this step
    temp_index = parameters.index('Temperature')
    this_temp = round(float(step_data[2].split('\t')[temp_index]))

    # parse step time and modulus data
    time_index = parameters.index('Step time')
    modulus_index = parameters.index('Modulus')

    # determine filename of oputput file
    output_filename = f'{filename[:-4]}_step{step_num}_{this_temp}.txt'

    # write output file
    with open(output_filename, 'w') as f:
        for pnt in step_data[3:]:
            pnt = pnt.split('\t')
            if float(pnt[time_index]) >= cutoff_time:
                t = float(pnt[time_index])
                m = float(pnt[modulus_index])
                f.write(f'{t}\t{m}\n')

    print(f'Step {step_num} | {this_temp} °C | '
          f'written to {output_filename}')


# --- MAIN --------------------------------------------------------------------
stress_relaxation = get_stressrelaxation(filename)

for i, step in enumerate(stress_relaxation, start=1):
    parse_stressrelaxation(step, cutoff_time, i)
