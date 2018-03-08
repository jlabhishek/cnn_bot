import os, shutil
import pickle
folder = 'save_model'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
    	pass
# pickle.dump([], open('Files/rewards.pickle', 'wb'))
pickle.dump([], open('Files/steps.pickle', 'wb'))
open('Files/q_action_table.txt', 'w').close()
open('Files/q_val_table.txt', 'w').close()
# open('Files/summary.txt', 'w').close()
# open('Files/deep_q_table.txt', 'w').close()
