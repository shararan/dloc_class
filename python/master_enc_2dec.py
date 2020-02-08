
import os
from shutil import copyfile
from subprocess import call
import time

experiments = ["e19"]
sequence_of_exec = ["real2dec_train","real2dec_test","real2dec_disjoint_test"
					"real2dec_test_1env","real2dec_test_3env","real2dec_test_4env"]

params_dir = "./params_future/"

for i, e in enumerate(experiments):

	for s in sequence_of_exec:
		file_name = e + "_" + s + ".py"
		src = params_dir+file_name
		dst = "./params.py"
		if not os.path.isfile(src):
			continue
		print(src, dst)
		copyfile(src, dst)
		print("copied params, calling main on them")
		os.system('python main_enc_2dec.py')
		time.sleep(10)