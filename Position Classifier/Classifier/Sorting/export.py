import Image_extraction as ie

""" 
after importing the module all we have to do is
to specify which hospital and the folders path. 

"""
#Define input and output directories, hospital structure, and info file
input_dir = "C:/Users/student/Desktop/images/processed_images/Dresden_processed/armsUp"
output_dir = "C:/Users/student/Desktop/images/Module_test/Dresden/armsUp"
hospital_structure = "Dresden"
info_file = "E:/Jasper/DBCGRT/Dataset/Dresden CGC/BCCT HYPO Dresden CGC 03.2018.xlsx"

ie.sort_and_rename_images(input_dir, output_dir,
                          info_file, hospital_structure)
