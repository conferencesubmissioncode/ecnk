from subprocess import call, Popen
import sys
server_id = int(sys.argv[1])
for gpu in range(8 * server_id, 8 * server_id + 8):
	command = ("screen python main.py -l tangent -ngpu 40 -lgpu 8 -gpu " + str(gpu) + " -d 13 -d_delta 3 -d_min 4 -c 20 -c_delta 4 -c_min 4 -dir kernel_full -bsize 100 -pixel 28 -ker_num 2 -f yes").split()
	print command
	Popen(command)

