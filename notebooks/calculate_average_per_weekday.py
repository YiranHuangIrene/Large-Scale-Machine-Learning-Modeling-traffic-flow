# %%
from datetime import datetime
import numpy as np
import torch
import h5py
from pathlib import Path
from skimage.measure import block_reduce
from tqdm import tqdm
from matplotlib import pyplot as plt

ROOT_PATH = Path("/nfs/shared/traffic4cast/")
CITIES = ["ANTWERP", "BANGKOK", "BARCELONA", "BERLIN", "CHICAGO", "ISTANBUL", "MELBOURNE", "MOSCOW", "NEWYORK", "VIENNA"]

# %%
# %matplotlib inline

# %%
def get_weekday_from_path(path):
    filename = path.name.split("/")[-1]
    date_string = filename[:10]
    date_object = datetime.strptime(date_string, "%Y-%m-%d").date()
    return date_object.weekday()


# %%
def extract_h5_data(path:str):
    """
    this function extracts numpy arrays from h5 files
    @param path: string with path to file that shall be extracted
    @return: numpy.array with data
    """
    hf = h5py.File(path, 'r')
    # kf_keys = [key for key in hf.keys()]
    # data = [torch.tensor(hf.get(key)) for key in kf_keys]
    data = hf['array']
    data = np.array(data)
    hf.close()
    return data

# %%
directory = (ROOT_PATH / "MELBOURNE")/ "training"

# %%
files = [f for f in directory.rglob("*.h5")]
print(f"Found {len(files)} files")

# %%
per_day_files = {i:[] for i in range(7)}
for f in files:
    per_day_files[get_weekday_from_path(f)].append(f) 

# %%
print(" Files per day: "+str({i: len(per_day_files[i]) for i in per_day_files.keys()}))

# %%
# print(f"Load file {files[0]}")
# ex_file = extract_h5_data(files[0])

# %%
# print(len(ex_file))
# print(ex_file.shape)
# print(ex_file.dtype)

# %%
# block_reduce(ex_file, block_size=(12, 1, 1, 1), func=np.mean).shape

# %%
per_day_avg =  []

# %%
# for day, day_files in tqdm(per_day_files.items(), total=len(per_day_files), leave=False):
#     per_day_images = []
#     for n, file in tqdm(enumerate(day_files), total=len(day_files), leave=False):
#         file_data = extract_h5_data(file)
#         per_day_images.append(file_data)
#     per_day_images = np.array(per_day_images)
#     daily_mean = np.mean(per_day_images[per_day_images.nonzero()], axis=0)
#     daily_mean_uint8 = daily_mean.astype('uint8')
#     per_day_avg.append(daily_mean_uint8)
# del per_day_images


for day, day_files in tqdm(per_day_files.items(), total=len(per_day_files), leave=False):
    per_day_sum = np.zeros((288,495,436,8), dtype=np.uint8)
    per_day_activity = np.zeros((288,495,436,8), dtype=np.uint8)
    for n, file in tqdm(enumerate(day_files), total=len(day_files), leave=False):
        file_data = extract_h5_data(file)
        per_day_sum += (file_data)
        per_day_activity += (file_data!=0)
    daily_mean = np.divide(per_day_sum, per_day_activity, where=per_day_activity!=0)
    daily_mean_uint8 = daily_mean.astype('uint8')
    per_day_avg.append(daily_mean_uint8)
del per_day_activity
del per_day_sum

# %%
# per_day_avg = {k: np.mean(v, axis=0) for k,v in per_day_images.items()}
np.save( "/nfs/shared/traffic4cast/MELBOURNE_traffic_per_weekday_masked.npy", per_day_avg)
# torch.save(torch.from_numpy(per_day_avg), "/nfs/shared/traffic4cast/MELBOURNE_traffic_per_weekday.pt")

# %%
# days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# %%
# fig = plt.figure(figsize=(14,2))
# for day, img in per_day_avg.items():
#     daily_average = block_reduce(img, block_size=(288, 1, 1, 8), func=np.mean).squeeze()
#     ax = fig.add_subplot(1, 7, day+1)
#     ax.imshow(daily_average, cmap='magma')
#     ax.set_title(f"{days[day]}")
#     ax.set_axis_off()
# fig.show()

# %%
# plt.imshow(per_day_avg[0][140,:,:,1].squeeze(), cmap='magma')
# plt.show()

# %%
# daily_average = block_reduce(per_day_avg[0], block_size=(12, 1, 1, 8), func=np.mean).squeeze()
# fig = plt.figure(figsize=(2, 48))
# for hour in range(daily_average.shape[0]):
#     ax = fig.add_subplot(24, 1, hour+1)
#     ax.imshow(daily_average[hour], cmap='magma')
#     ax.set_title(f"{hour}:00-{hour}:59")
#     ax.set_axis_off()
# fig.show()

# %%
# traffic_per_week = torch.load("traffic_per_weekday.pt")

# %%
# traffic_per_week[0].shape

# %%
# plt.imshow(traffic_per_week[0][0,:,:,0], cmap="gist_stern")

# %%



