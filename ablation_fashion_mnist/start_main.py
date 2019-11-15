from subprocess import call, Popen
import sys
server_id = int(sys.argv[1])
for gpu in range(8 * server_id, 8 * server_id + 8):
	command = ("screen python main.py -ngpu 40 -lgpu 8 -gpu " + str(gpu) + " -d 13 -d_delta 3 -d_min 4 -c 28 -c_delta 4 -c_min 0 -dir kernel_full -f yes -bsize 250 -pixel 28 -ker_num 2 -fargs {\"dataset\":\"fashion_mnist\"}").split()
	print command
	Popen(command)

