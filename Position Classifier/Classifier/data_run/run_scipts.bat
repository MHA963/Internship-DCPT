@echo off

echo Running Aalborg dataset 
c:/Users/student/anaconda3/envs/yolo_test_env/python.exe data_run_aal.py > Aalborg.txt

echo Running Odense dataset...
c:/Users/student/anaconda3/envs/yolo_test_env/python.exe data_run_Ode.py > Odense.txt

echo Running Vejle Dataset...
c:/Users/student/anaconda3/envs/yolo_test_env/python.exe data_run_vej.py > Vejle.txt

echo Running Aarhus Dataset... 
c:/Users/student/anaconda3/envs/yolo_test_env/python.exe data_run_aar.py > Aarhus.txt

echo All scripts completed.
