import subprocess

# Define the first script to run
script1 = "C:/Users/student/Desktop/images/scale_marker/scale_marker.py"
script2 = "C:/Users/student/Desktop/images/scale_model/model_training.py"
script3 = "C:/Users/student/Desktop/images/scale_marker/model_test.py"    
# Define the three scripts to run
scripts = [script1, script2, script3]

for script in scripts:
    result = subprocess.run(["C:/Users/student/anaconda3/envs/yolo_test_env/python.exe", script])
    
    if result.returncode != 0:
        print(f"Script {script} did not execute successfully. Exiting.")
        break

print("All scripts have been executed.")