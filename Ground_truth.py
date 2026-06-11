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


############################# CPU ##############################
# print("1. 正在读取 SIFT1M Base 数据...")
# base_data = read_fvecs("sift_learn.fvecs")
# N, dim = base_data.shape
#
# print("2. 正在构建 FAISS IndexFlatL2 并寻找 top-101 近邻...")
# # 使用暴力精确搜索 (L2距离)
# index = faiss.IndexFlatL2(dim)
# index.add(base_data)
#
# # 搜索前 101 个邻居 (因为第 1 个肯定是自己，距离为 0)
# # 如果你有 GPU，可以用 faiss.index_cpu_to_gpu 加速，几秒钟搞定
# distances, indices = index.search(base_data, 101)
#
# print("3. 剔除自身节点并保存为 ivecs 格式...")
# # 去掉每一行的第 0 列 (自身节点)
# knn_graph = indices[:, 1:101].astype(np.int32)
#
# write_ivecs("sift_learn_100nn.ivecs", knn_graph)
# print("完成！已生成 sift_100nn.ivecs 文件。")


############################# CPU ##############################
# print("1. 正在读取 MNIST 训练集数据...")
# mnist_data = read_idx3_ubyte("train-images.idx3-ubyte")
# N, dim = mnist_data.shape
# print(f"数据形状: {N} x {dim}")
#
# print("2. 构建 FAISS CPU 索引 (IndexFlatL2)...")
# index = faiss.IndexFlatL2(dim)
# index.add(mnist_data)
#
# print("3. 搜索每个向量的 top-101 近邻...")
# distances, indices = index.search(mnist_data, 101)
#
# print("4. 剔除自身节点并保存为 ivecs 格式...")
# knn_graph = indices[:, 1:101].astype(np.int32)
# write_ivecs("mnist_train_100nn.ivecs", knn_graph)
# print("完成！已生成 mnist_train_100nn.ivecs 文件。")

############################# CPU ##############################
# print("1. 正在读取 Deep1M 训练集数据 (base.1M.fbin)...")
# deep_data = read_fbin("base.1M.fbin")   # 请确保文件路径正确
# N, dim = deep_data.shape
# print(f"数据形状: {N} x {dim}")
#
# print("2. 构建 FAISS CPU 索引 (IndexFlatL2)...")
# index = faiss.IndexFlatL2(dim)
# index.add(deep_data)
#
# print("3. 搜索每个向量的 top-101 近邻...")
# distances, indices = index.search(deep_data, 101)
#
# print("4. 剔除自身节点并保存为 ivecs 格式...")
# knn_graph = indices[:, 1:101].astype(np.int32)
# write_ivecs("deep1m_100nn.ivecs", knn_graph)
# print("完成！已生成 deep1m_100nn.ivecs 文件。")

# ############################# CPU ##############################
# N = 1000000          # 向量总数
# D = 256                # 维度
# print(f"正在生成 {N} x {D} 的随机浮点数据 (float32)...")
# # 生成 [0,1) 均匀分布的随机数，与 SIFT 的像素值分布近似
# random_data = np.random.rand(N, D).astype(np.float32)
# print(f"数据形状: {random_data.shape}, 内存占用: {random_data.nbytes / 1024**3:.2f} GB")
#
# # 可选：保存为 fvecs 文件，方便后续使用
# fvecs_file = "random_1M_256D.fvecs"
# print(f"正在保存为 fvecs 文件: {fvecs_file}")
# write_fvecs(fvecs_file, random_data)
# print("保存完成。")
#
# # ----------------------------- 3. 构建 FAISS 索引并搜索 101-NN -----------------------------
# print("构建 FAISS CPU 索引 (IndexFlatL2)...")
# index = faiss.IndexFlatL2(D)
# index.add(random_data)
#
# k = 101  # 搜索 101 个近邻（第 1 个是自身）
# print(f"正在搜索每个向量的 top-{k} 近邻...")
# # 注意：搜索 100 万 × 101 的结果矩阵内存较大，约 1e6*101*4=404 MB
# distances, indices = index.search(random_data, k)
#
# # ----------------------------- 4. 去除自身并保存为 ivecs -----------------------------
# print("剔除自身节点（每行第一个索引）...")
# knn_graph = indices[:, 1:k].astype(np.int32)  # 取第 2 到第 101 列，共 100 个邻居
#
# output_file = "random_1M_256D_100nn.ivecs"
# print(f"保存结果到 {output_file} ...")
# write_ivecs(output_file, knn_graph)
# print("完成！")


############################ CPU ##############################
print("1. 正在读取 GIST1M 训练集数据 (gist_base.fvecs)...")
base_data = read_fvecs("gist_base.fvecs")
N, dim = base_data.shape
print(f"数据形状: {N} x {dim}")

print("2. 构建 FAISS CPU 索引 (IndexFlatL2)...")
index = faiss.IndexFlatL2(dim)
index.add(base_data)

print("3. 搜索每个向量的 top-101 近邻...")
distances, indices = index.search(base_data, 101)

print("4. 剔除自身节点并保存为 ivecs 格式...")
knn_graph = indices[:, 1:101].astype(np.int32)
write_ivecs("gist1m_100nn.ivecs", knn_graph)
print("完成！已生成 gist1m_100nn.ivecs 文件。")
