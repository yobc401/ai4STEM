from functools import partial
from ai4STEM.utils.utils_config import set_configs
from ai4STEM.dataprocessing.preprocessing import load_dataset_from_file
from ai4STEM.models.cnn_architectures import cnn_nature_comm_ziletti2018
from ai4STEM.models.cnn_STEM_model import load_datasets
from ai4STEM.models.STEM_CNN_segmentation import predict_value # YBC
from ai4STEM.models.STEM_CNN_segmentation import predict_with_uncertainty # YBC
from ai4STEM.models.cnn_STEM_model import train_neural_network
from ai4STEM.utils.utils_config import setup_logger
from sklearn import preprocessing
import numpy as np
import os

from ai4STEM.utils.utils_crystals import get_boxes_from_xyz

import time


configs = set_configs()
logger = setup_logger(configs, level='DEBUG', display_configs=False)
#dataset_folder = configs['io']['main_folder']
dataset_folder = os.path.join(configs['io']['main_folder'], 'my_datasets')

# =============================================================================
# Download the dataset from the online repository and load it
# =============================================================================

#x_pristine, y_pristine, dataset_info_pristine, x_vac25, y_vac25, dataset_info_vac25 = load_datasets(dataset_folder)

test_set_name = 'STEM_monocrystalline'
path_to_x_box = os.path.join(dataset_folder, test_set_name + '_x.pkl')
path_to_y_box = os.path.join(dataset_folder, test_set_name + '_y.pkl')
path_to_summary_box = os.path.join(dataset_folder, test_set_name + '_summary.json')


x_box, y_box, dataset_info_box = load_dataset_from_file(path_to_x_box, path_to_y_box,
                                                              path_to_summary_box)

#print(x_pristine)

# =============================================================================
# Train the convolutional neural network
# =============================================================================

# load the convolutional neural network architecture from Ziletti et al., Nature Communications 9, pp. 2775 (2018)
#partial_model_architecture = partial(cnn_nature_comm_ziletti2018, conv2d_filters=[32, 32, 16, 16, 8, 8],
#                                     kernel_sizes=[3, 3, 3, 3, 3, 3], max_pool_strides=[2, 2],
#                                     hidden_layer_size=128)

# use x_train also for validation - this is only to run the test
#results = train_neural_network(x_train=x_pristine, y_train=y_pristine, x_val=x_pristine, y_val=y_pristine,
#                               configs=configs, partial_model_architecture=partial_model_architecture,
#                               nb_epoch=10)


f_list = open("list.txt",'r')

targets = []

while True:

    line = f_list.readline()

    if not line:
        break


    cols = line.split()

    targetname = 'fcc111'#cols[1].rstrip()
    targets.append(targetname)



label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(targets)
labels_list = label_encoder.transform(targets)


text_labels = np.array(targets) #label_encoder.classes_ 
numerical_labels = label_encoder.transform(targets)

print(len(numerical_labels))

# =============================================================================
# Predict the crystal class of a material using the trained neural network
# =============================================================================

# load the convolutional neural network architecture from Ziletti et al., Nature Communications 9, pp. 2775 (2018)
# you can also use your own neural network to predict, passing it to the variable 'model'
y_pred = predict_value(x_box, y_box, configs=configs, numerical_labels=numerical_labels,
                  text_labels=text_labels, nb_classes = 11,  model=None, show_model_acc=False)  #YBC nb_classes
print(len(y_pred))
print(y_pred)


class_list = [["bcc100"],["bcc110"],["bcc111"],["fcc100"],["fcc110"],["fcc111"],["fcc211"],["hcp0001"],["hcp10m10"],["hcp11m20"],["hcp2m1m10"]]

"""
filename = 'polycrystal.xyz'

frame_size=[112.0, 112.0, 80.0]
sliding_volume=[14.0, 14.0, 80.0]
stride_size=[7.0, 7.0, 80.0]
start_time = time.time()
boxes, number_of_atoms_xyz = get_boxes_from_xyz(filename,
                                                frame_size=frame_size,
                                                sliding_volume=sliding_volume,
                                                stride_size= stride_size,
                                                adapt=False,
                                                give_atom_density=True,
                                                plot_atom_density=False,
                                                padding_ratio=0.0)

y_grid_max = len(boxes[0])
x_grid_max = len(boxes[0][0])
"""
y_grid_max = 5
x_grid_max = 5
print(x_grid_max)
#x = 0
#y = 0
#convert_from_boxes_to_input(boxes[0][y][x],x,y,sliding_volume[0],sliding_volume[1],sliding_volume[2])

fmap = open("map.txt",'w')

idx = 0

for y in range(y_grid_max):
    for x in range(x_grid_max):
            
        fmap.write("%10s" %(class_list[y_pred[idx]][0]))
        fmap.write(" ")
        idx = idx + 1

    fmap.write("\n")



fmap.close()



num_index = x_grid_max * y_grid_max
prediction, uncertainty = predict_with_uncertainty(x_box[0:num_index+1])

print("prediction")
print(prediction)



print("uncertainty")
print(uncertainty)

list_variation_ratio = uncertainty['variation_ratio']
list_predictive_entropy = uncertainty['predictive_entropy']
list_mutual_information = uncertainty['mutual_information']


fvr = open("var_rat.txt",'w')
idx = 0
for y in range(y_grid_max):
    for x in range(x_grid_max):
        fvr.write("%10.4f" %(list_variation_ratio[idx]))
        fvr.write(" ")
        idx = idx + 1
    fvr.write("\n")
fvr.close()


fpe = open("pred_ent.txt",'w')
idx = 0
for y in range(y_grid_max):
    for x in range(x_grid_max):
        fpe.write("%10.4f" %(list_predictive_entropy[idx]))
        fpe.write(" ")
        idx = idx + 1
    fpe.write("\n")
fpe.close()


fmi = open("mut_inf.txt",'w')
idx = 0
for y in range(y_grid_max):
    for x in range(x_grid_max):
        fmi.write("%10.4f" %(list_mutual_information[idx]))
        fmi.write(" ")
        idx = idx + 1
    fmi.write("\n")
fmi.close()


