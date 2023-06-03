import tick
from tick.hawkes import SimuHawkes, HawkesKernelExp
from tqdm.auto import tqdm
import numpy as np
import torch


def generate_points(mu, alpha, decay, window, seed, dt=0.01):
    """
    Generates points of an marked Hawkes processes using the tick library
    """

    n_processes = len(mu)
    hawkes = SimuHawkes(n_nodes=n_processes, end_time=window, verbose=False, seed=seed)
    
    for i in range(n_processes):
        for j in range(n_processes):
            hawkes.set_kernel(i=i, j=j, kernel=HawkesKernelExp(intensity=alpha[i][j] / decay[i][j], decay=decay[i][j]))
        hawkes.set_baseline(i, mu[i])
        
    hawkes.track_intensity(dt)
    hawkes.simulate()
    return hawkes.timestamps


def hawkes_helper(mu, alpha, decay, window, in_seed, in_range):
    times_marked = [generate_points(mu=mu, alpha=alpha, decay=decay, window=window, seed=in_seed + i) for i in
                    tqdm(range(in_range))]
    records = [hawkes_seq_to_record(r) for r in (times_marked)]
    return records


def hawkes_seq_to_record(seq):
    times = np.concatenate(seq)
    labels = np.concatenate([[i] * len(x) for i, x in enumerate(seq)])
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    labels = labels[sort_idx]
    record = [
        {"time": float(t),
         "labels": (int(l),)} for t, l in zip(times, labels)]
    return record


def combine_splits(d_train, d_val, d_test):
    sequences = []

    for dataset in ([d_train, d_val, d_test]):
        for i in range(len(dataset)):
            event_dict = {}
            arrival_times = []
            marks = []
            for j in range(len(dataset[i])):
                curr_time = dataset[i][j]['time']
                curr_mark = dataset[i][j]['labels'][0]
                arrival_times.append(curr_time)
                marks.append(curr_mark)

            event_dict['t_start'] = 0
            event_dict['t_end'] = 100
            event_dict['arrival_times'] = arrival_times
            event_dict['marks'] = marks

            sequences.append(event_dict)

    return sequences


def dataset_helper(mu, alpha, beta, window, seed, train_size, val_size, test_size, save_path):
    train_seed = seed
    val_seed = seed + train_size
    test_seed = seed + train_size + val_size

    d_train = hawkes_helper(mu, alpha, beta, window, train_seed, train_size)
    d_val = hawkes_helper(mu, alpha, beta, window, val_seed, val_size)
    d_test = hawkes_helper(mu, alpha, beta, window, test_seed, test_size)

    sequences = combine_splits(d_train, d_val, d_test)
    dataset = {'sequences': sequences, 'num_marks': len(mu)}
    torch.save(dataset, save_path)





















































torch.manual_seed(0)
import torch
m=3
mu=torch.rand(m).tolist()
print(mu)


alpha=torch.rand(m,m).tolist()



print(alpha)
beta=torch.clamp(10*torch.rand(m,m),1).tolist()



print(beta)
window = 50
seed = 0
size=3000
train_size = int(size*0.6)
val_size = int(size*0.2)
test_size = int(size*0.2)
save_path = '../data/hawkes_dep_m6.pkl'

dataset_helper(mu, alpha, beta, window, seed, train_size, val_size, test_size, save_path)
