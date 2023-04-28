from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
data = [1,2,3,4]
if rank == 0:
    data[3]=1
    comm.send(data, dest=1, tag=11)
elif rank == 1:
    data = comm.recv(source=0, tag=11)
    data[2]=1
print(data)
   