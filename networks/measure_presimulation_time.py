import os
import numpy as np


def average_simulation_time(folder_path: str, key_words: str, cores: int = 8):
    total_time = 0
    count = 0
    for filename in os.listdir(folder_path):
        if 'DS_Store' in filename:
            continue
        with open(os.path.join(folder_path, filename), 'r') as file:
            for line in file:
                if key_words in line:
                    text = line.split(":")[1].strip().split(" ")[0]
                    time = float(text)
                    total_time += time
                    count += 1

    print('average per run (min):',  np.round(total_time / count, 2))
    print('total (h):', np.round(total_time / 60, 2))
    print('using multi-processing total (h):', np.round(total_time / 60 / cores, 2))
    return


model_names = ['fröhlich_detailed', 'fröhlich_sde', 'pharma', 'clairon', 'clairon uniform']
print('simulation time')
for model_name in model_names:
    print(model_name)
    path = os.path.join(os.getcwd(), 'log presimulation/' + model_name)
    average_simulation_time(path, key_words='simulation time')
    print('\n')


model_names = ['fröhlich_simple', 'fröhlich_detailed', 'fröhlich_sde', 'pharma', 'clairon', 'clairon uniform']
print('\n\ntraining time')
for model_name in model_names:
    print(model_name)
    path = os.path.join(os.getcwd(), 'log training/' + model_name)
    average_simulation_time(path, key_words='training time')
    print('\n')

# simulation & training time using 8 cores
# fröhlich_simple: 0 h + 6.11 h
# fröhlich_detailed: 3.74 h + 8.0 h
# fröhlich_sde: 2.16 h + 5.12 h
# pharma: 9.83 h +  11.25 h
# clairon normal: 3.33 h + 2.62 h
# clairon uniform: 2.97 h + 3.21 h

