from ase.spacegroup import crystal
from ai4materials.dataprocessing.preprocessing import load_dataset_from_file
from ai4materials.dataprocessing.preprocessing import prepare_dataset_STEM  ### YBC
#from ai4materials.descriptors.diffraction2d import Diffraction2D
from ai4materials.utils.utils_config import set_configs
from ai4materials.utils.utils_config import setup_logger
from ai4materials.utils.utils_crystals import create_supercell
from ai4materials.utils.utils_crystals import create_vacancies
from ai4materials.wrappers import calc_descriptor
from ai4materials.wrappers import load_descriptor
import os.path
from sklearn import preprocessing
import numpy as np
from PIL import Image


# set configs
configs = set_configs(main_folder='./')
logger = setup_logger(configs, level='INFO', display_configs=False)

# setup folder and files
dataset_folder = os.path.join(configs['io']['main_folder'], 'my_datasets')
desc_file_name = 'fcc_bcc_hcp_example'


# calculate the descriptor for the list of structures 
images_list = []
targets_list = []


f_list = open("list_shuf_test.txt",'r')
while True:
    line = f_list.readline()

    if not line:
        break

    cols = line.split()
    

    filename = "CBED/" + cols[0]
    filename = filename.rstrip()

    targetname = cols[1].rstrip()
    print(filename)
    im = Image.open(filename)
    size = (64,64)
    im.thumbnail(size)
    output = np.array(im,dtype=float)

    np_image = np.zeros((64,64,3),'uint8')
    for x in range(0,64):
        for y in range(0,64):
            for ch in range(0,3):
                np_image[x,y,ch] = output[x,y,ch] #* 255 

    images_list.append(np_image)
    targets_list.append(targetname)




"""

f_image_list = open("image_list_part.txt",'r')
f_target_list = open("target_list_part.txt",'r')

while True:

    line1 = f_image_list.readline()
    line2 = f_target_list.readline()

    if not line1:
        break
    if not line2:
        break


    filename = "/home/yeo/Work/ai4STEM/prototype/SimSTEM_DB_2/FFT/" + line1
    filename = filename.rstrip()

    targetname = line2.rstrip()
    print(filename)
    im = Image.open(filename)
    size = (64,64)
    im.thumbnail(size)
    output = np.array(im,dtype=float)

    np_image = np.zeros((64,64,3),'uint8')
    for x in range(0,64):
        for y in range(0,64):
            for ch in range(0,3):
                np_image[x,y,ch] = output[x,y,ch] #* 255 

    images_list.append(np_image)
    targets_list.append(targetname)


"""

images_list = np.array(images_list)



# add as target the spacegroup (using spacegroup of the "parental" structure for the defective structure)
#targets = ['fcc111', 'fcc111', 'fcc110', 'fcc110', 'fcc100', 'fcc100']

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(targets_list)
labels_list = label_encoder.transform(targets_list)



print(images_list.shape)
print(labels_list.shape)


path_to_x, path_to_y, path_to_summary = prepare_dataset_STEM(
    images_list=images_list,
    labels_list=labels_list,
    desc_metadata='diffraction_2d_intensity',
    dataset_name='STEM_monocrystalline',
    target_name='target',
    target_categorical=True,
    input_dims=(64, 64),
    configs=configs,
    dataset_folder=dataset_folder,
    main_folder=configs['io']['main_folder'],
    desc_folder=configs['io']['desc_folder'],
    tmp_folder=configs['io']['tmp_folder'],
    notes="STEM Dataset with bcc, fcc, and hcp structures, pristine.")

x, y, dataset_info = load_dataset_from_file(path_to_x=path_to_x, path_to_y=path_to_y,
                                                              path_to_summary=path_to_summary)

#print(x)

