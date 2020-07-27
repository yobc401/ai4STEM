from ai4materials.utils.utils_crystals import get_boxes_from_xyz

import os, time


filename = 'polycrystal.xyz'

frame_size=[112.0, 112.0, 80.0]
sliding_volume=[14.0, 14.0, 80.0]
stride_size=[14.0, 14.0, 80.0]
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


pixelSizeX = 0.1
pixelSizeY = 0.1

nX = 32
nY = 32


#x = 0
#y = 0
#convert_from_boxes_to_input(boxes[0][y][x],x,y,sliding_volume[0],sliding_volume[1],sliding_volume[2])


f_list = open("list.txt","w")

for y in range(y_grid_max):
    for x in range(x_grid_max):
        if len(boxes[0][y][x]) != 0:
            #convert_from_boxes_to_input(boxes[0][y][x],x,y,sliding_volume[0],sliding_volume[1],sliding_volume[2])
            #os.system("mv Input*.XYZ INPUT/.")
            #os.system("mv POSCAR*.vasp POSCAR/.")
 
            label = str(x) + "_" +  str(y)

            f_list.write("CBEDimage_"+label+".png"+"\t"+"fcc111"+"\n")

            

            """

            os.system("cp INPUT/Input_" + label + ".XYZ Input.XYZ")


            sim_prismatic.sim_prismatic(pixelSizeX, pixelSizeY)
            sim_image.sim_HAADF(label)
            sim_image.sim_fft(label,nX,nY)
            os.system("mv HAADF*.png HAADF/.")
            os.system("mv FFT*.png FFT/.")
            os.system("rm -f prismatic*.mrc")

            """

f_list.close()

end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))
