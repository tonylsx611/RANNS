import numpy as np
import faiss
import struct

def read_fvecs(filename):
    with open(filename, 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
        f.seek(0, 2)
        filesize = f.tell()
        num = filesize // (4 + 4 * dim)
        f.seek(0)
        a = np.fromfile(f, dtype=np.int32)
        a = a.reshape(-1, 1 + dim)
        return a[:, 1:].copy().view(np.float32)

def read_fbin(filename):
    with open(filename, 'rb') as f:
        num, dim = struct.unpack('II', f.read(8))   # 'II' 表示两个 unsigned int (小端序)
        data = np.fromfile(f, dtype=np.float32)
        return data.reshape(num, dim)

def read_idx3_ubyte(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError("Not a valid MNIST image file")
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
        return images.astype(np.float32)

def write_fvecs(filename, data):
    with open(filename, 'wb') as f:
        for row in data:
            dim = np.array([len(row)], dtype=np.int32)
            dim.tofile(f)
            row.astype(np.float32).tofile(f)

def write_ivecs(filename, array):
    with open(filename, 'wb') as f:
        for row in array:
            dim = np.array([len(row)], dtype=np.int32)
            dim.tofile(f)
            row_int = np.array(row, dtype=np.int32)
            row_int.tofile(f)


###########################################################
print("1. Loading SIFT1M_Base Dataset...")
base_data = read_fvecs("sift_learn.fvecs")
N, dim = base_data.shape

if base_data.dtype != np.float32:
    base_data = base_data.astype(np.float32)

res = faiss.StandardGpuResources()  # Faiss GPU

print("2. Constructing FAISS IndexFlatL2 and discover 101-NN...")
cpu_index = faiss.IndexFlatL2(dim)
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
gpu_index.add(base_data)

distances, indices = gpu_index.search(base_data, 101)

print("3. Save as .ivecs Format...")
knn_graph = indices[:, 1:101].astype(np.int32)

write_ivecs("sift_learn_100nn.ivecs", knn_graph)
print("Completely Generating file: sift_100nn.ivecs")


###########################################################
print("1. Loading MNIST Dataset...")
mnist_data = read_idx3_ubyte("train-images.idx3-ubyte")
N, dim = mnist_data.shape

if mnist_data.dtype != np.float32:
    mnist_data = mnist_data.astype(np.float32)

res = faiss.StandardGpuResources()  # Faiss GPU

print("2. Constructing FAISS IndexFlatL2 and discover 101-NN...")
cpu_index = faiss.IndexFlatL2(dim)
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
gpu_index.add(mnist_data)

distances, indices = gpu_index.search(mnist_data, 101)

print("3. Save as .ivecs Format...")
knn_graph = indices[:, 1:101].astype(np.int32)

write_ivecs("mnist_train_100nn.ivecs", knn_graph)
print("Completely Generating file: mnist_train_100nn.ivecs")

###########################################################
print("1. Loading Deep1M Dataset...")
deep_data = read_fbin("base.1M.fbin")
N, dim = deep_data.shape

if deep_data.dtype != np.float32:
    deep_data = deep_data.astype(np.float32)

res = faiss.StandardGpuResources()  # Faiss GPU

print("2. Constructing FAISS IndexFlatL2 and discover 101-NN...")
cpu_index = faiss.IndexFlatL2(dim)
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
gpu_index.add(deep_data)

# Safe batch search to prevent GPU OOM
indices = np.zeros((N, 101), dtype=np.int64)
batch_size = 50000
for i in range(0, N, batch_size):
    end = min(i + batch_size, N)
    _, batch_indices = gpu_index.search(deep_data[i:end], 101)
    indices[i:end] = batch_indices

print("3. Save as .ivecs Format...")
knn_graph = indices[:, 1:101].astype(np.int32)

write_ivecs("deep1m_100nn.ivecs", knn_graph)
print("Completely Generating file: deep1m_100nn.ivecs")


###########################################################
print("1. Loading GIST1M Dataset...")
base_data = read_fvecs("gist_base.fvecs")
N, dim = base_data.shape

if base_data.dtype != np.float32:
    base_data = base_data.astype(np.float32)

res = faiss.StandardGpuResources()  # Faiss GPU

print("2. Constructing FAISS IndexFlatL2 and discover 101-NN...")
cpu_index = faiss.IndexFlatL2(dim)
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
gpu_index.add(base_data)

# Safe batch search to prevent GPU OOM
indices = np.zeros((N, 101), dtype=np.int64)
batch_size = 50000
for i in range(0, N, batch_size):
    end = min(i + batch_size, N)
    _, batch_indices = gpu_index.search(base_data[i:end], 101)
    indices[i:end] = batch_indices

print("3. Save as .ivecs Format...")
knn_graph = indices[:, 1:101].astype(np.int32)

write_ivecs("gist1m_100nn.ivecs", knn_graph)
print("Completely Generating file: gist1m_100nn.ivecs")
