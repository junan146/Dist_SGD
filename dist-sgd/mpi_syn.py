from mpi_worker import Worker
from myutils import Timer
from mpi4py import MPI
from model_simple import SimpleCNN

import mpi_dataset

import logging



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    comm = MPI.COMM_WORLD
    worker_index = comm.Get_rank()
    worker_size = comm.Get_size()


    QuantizerWorker = Worker

    if worker_index == 0:
        logging.info('initialize quantization worker as {}'.format(QuantizerWorker))

    dataset = mpi_dataset.download_mnist_retry(seed=worker_index)

    nn = SimpleCNN(dataset)

    worker = QuantizerWorker(net=nn, dataset=dataset, lr=1e-4)

    worker.syn_weights(worker.net.variables.get_flat())

    if worker_index == 0:
        logging.info("Iteration, time, loss, accuracy")

    timer = Timer()
    i = 0
    while i <= 2000:
        if i % 10 == 0:
            # Evaluate the current model.
            loss, accuracy = worker.compute_loss_accuracy()
            if worker_index == 0:
                logging.info("%d, %.3f, %.3f, %.3f" % (i, timer.toc(), loss, accuracy))
        i += 1
        worker.shuffle_reduce(worker.compute_gradients())
