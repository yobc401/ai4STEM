from functools import partial
from ai4materials.utils.utils_config import set_configs
from ai4materials.dataprocessing.preprocessing import load_dataset_from_file
from ai4materials.models.cnn_architectures import cnn_nature_comm_ziletti2018
from ai4materials.models.cnn_nature_comm_ziletti2018 import load_datasets
from ai4materials.models.STEM_CNN_segmentation import predict # YBC
from ai4materials.models.cnn_nature_comm_ziletti2018 import train_neural_network
from ai4materials.utils.utils_config import setup_logger
from sklearn import preprocessing
import numpy as np
import os

configs = set_configs()
logger = setup_logger(configs, level='DEBUG', display_configs=False)
#dataset_folder = configs['io']['main_folder']
dataset_folder = os.path.join(configs['io']['main_folder'], 'my_datasets')

# =============================================================================
# Download the dataset from the online repository and load it
# =============================================================================

#x_pristine, y_pristine, dataset_info_pristine, x_vac25, y_vac25, dataset_info_vac25 = load_datasets(dataset_folder)

train_set_name = 'STEM_monocrystalline_test'
path_to_x_pristine = os.path.join(dataset_folder, train_set_name + '_x.pkl')
path_to_y_pristine = os.path.join(dataset_folder, train_set_name + '_y.pkl')
path_to_summary_pristine = os.path.join(dataset_folder, train_set_name + '_summary.json')

test_set_name = 'STEM_monocrystalline_test'
path_to_x_vac25 = os.path.join(dataset_folder, test_set_name + '_x.pkl')
path_to_y_vac25 = os.path.join(dataset_folder, test_set_name + '_y.pkl')
path_to_summary_vac25 = os.path.join(dataset_folder, test_set_name + '_summary.json')

x_pristine, y_pristine, dataset_info_pristine = load_dataset_from_file(path_to_x_pristine, path_to_y_pristine,
                                                                       path_to_summary_pristine)

x_vac25, y_vac25, dataset_info_vac25 = load_dataset_from_file(path_to_x_vac25, path_to_y_vac25,
                                                              path_to_summary_vac25)

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


f_list = open("list_shuf_test.txt",'r')

targets = []

while True:

    line = f_list.readline()

    if not line:
        break


    cols = line.split()

    targetname = cols[1].rstrip()
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
results = predict(x_vac25, y_vac25, configs=configs, numerical_labels=numerical_labels,
                  text_labels=text_labels, nb_classes = 11,  model=None)  #YBC nb_classes

"""
class_list = [["bcc100"],["bcc110"],["bcc111"],["fcc100"],["fcc110"],["fcc111"],["fcc211"],["hcp0001"],["hcp10m10"],["hcp11m20"],["hcp2m1m10"]]


fmap = open("list_predict.txt",'w')

print(len(results))

for i in range(len(results)):
    #print(results)
    fmap.write("%10s" %(class_list[results[i]]))

fmap.close()


"""



