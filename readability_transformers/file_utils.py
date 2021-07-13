# Copyright 2021 One Theta. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import json
import requests
import shutil
import zipfile
import tarfile
import pickle
from tqdm import tqdm


DATA_CACHE_DIR = os.path.expanduser("~/.cache/readability-transformers/data")
MODEL_CACHE_DIR = os.path.expanduser("~/.cache/readability-transformers/models")
MAPPER_PATH = os.path.expanduser("~/.cache/readability-transformers/mapper.json")



def check_cache_exists_or_init():
    os.makedirs(os.path.expanduser('~/.cache/readability-transformers'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.cache/readability-transformers/data'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.cache/readability-transformers/data/all'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.cache/readability-transformers/data/preapply'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.cache/readability-transformers/models'), exist_ok=True)

    if not os.path.isfile(MAPPER_PATH):
        json.dump({"all": dict(), "preapply": dict(), "models": dict()}, open(MAPPER_PATH, "w"))
    else:
        mapper = load_dataset_mapper()
        if "all" not in mapper.keys():
            mapper["all"] = dict()
        elif "preapply" not in mapper.keys():
            mapper["preapply"] = dict()
        elif "models" not in mapper.keys():
            mapper["models"] = dict()
        save_dataset_mapper(mapper)
    
    return True

def download_file(url, dest):
    local_filename = url.split('/')[-1]
    dest = os.path.join(dest, local_filename)
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    print("Downloading:",url)
    with open(dest, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()    


    # with requests.get(url, stream=True) as r:
    #     with open(dest, 'wb') as f:
    #         shutil.copyfileobj(r.raw, f)

    return dest

def unzip_to(zip_file_path, target_path):
    if zip_file_path.endswith("tar.gz"):
        tar = tarfile.open(zip_file_path, "r:gz")
        tar.extractall(target_path)
        tar.close()
    elif zip_file_path.endswith("tar"):
        tar = tarfile.open(zip_file_path, "r:")
        tar.extractall(target_path)
        tar.close()
    elif zip_file_path.endswith("zip"):
        zip_ref = zipfile.ZipFile(zip_file_path, "r")
        zip_ref.extractall(target_path)
        zip_ref.close()

    return

def save_dataset_mapper(new_mapper):
    json.dump(new_mapper, open(MAPPER_PATH, "w"))
    return new_mapper

def load_dataset_mapper():
    mapper = json.load(open(MAPPER_PATH, "r"))
    return mapper

def load_from_cache_pickle(dataset_id, datafile_id):
    mapper = load_dataset_mapper()
    if dataset_id in mapper.keys():
        if datafile_id in mapper[dataset_id].keys():
            filepath_rel = mapper[dataset_id][datafile_id]
            filepath = os.path.join(DATA_CACHE_DIR, dataset_id, filepath_rel)
            data = pickle.load(open(filepath, "rb"))
            return data

    return None

def save_to_cache_pickle(dataset_id, datafile_id, filename, object):
    mapper = load_dataset_mapper()
    if dataset_id in mapper.keys():
        mapper[dataset_id][datafile_id] = filename
        with open(os.path.join(DATA_CACHE_DIR, dataset_id, filename), "wb") as f:
            pickle.dump(object, f)
        save_dataset_mapper(mapper)
    else:
        return False

    return mapper 
    

def check_mapper_or_download(dataset_id, datafile_id, dataset_url, datafiles_meta):
    mapper = load_dataset_mapper()
    if dataset_id not in mapper.keys():
        # Need to download.
        dest_url = os.path.join(DATA_CACHE_DIR, dataset_id)
        os.makedirs(dest_url, exist_ok=True)

        zipped_file_path = download_file(dataset_url, dest_url)
        unzip_to(zipped_file_path, dest_url)
        os.remove(zipped_file_path)
        
        mapper[dataset_id] = {}
        for dataname, datafile_path in datafiles_meta.items():
            if dataname is not "DATASET_ID":
                assert os.path.isfile(os.path.join(dest_url, datafile_path))
                mapper[dataset_id][dataname] = os.path.join(dest_url, datafile_path)
        save_dataset_mapper(mapper)
    else:
        if datafile_id not in mapper[dataset_id].keys():
            raise Exception(f"Requested non-existing datafile: {datafile_id}")
    
    return mapper

def path_to_rt_model_cache(model_name):
    mapper = load_dataset_mapper()
    model_mapper = mapper["models"]
    if model_name in model_mapper.keys():
        return model_mapper[model_name]
    else:
        return None

def download_rt_model(dataset_url):
    model_name = dataset_url.split("/")[-1].replace(".tar.gz", "")
    model_folder_path = os.path.join(MODEL_CACHE_DIR, model_name)

    print(f"Downloading model '{model_name}' from {dataset_url}...")
    zipped_file_path = download_file(dataset_url, MODEL_CACHE_DIR)
    unzip_to(zipped_file_path, model_folder_path)
    os.remove(zipped_file_path)

    folders = os.listdir(model_folder_path)
    actual_path = os.path.join(model_folder_path, folders[0])
    for one in glob.glob(actual_path + "/*"):
        shutil.move(one, os.path.join(model_folder_path, one.split("/")[-1]))
    os.rmdir(actual_path)


    # this actually also helps with mid-download errors since
    # if download & unzipping didn't 100% complete, the mapper wont have the info stored
    # which makes it download the whole thing again, which is what we want.
    mapper = load_dataset_mapper()
    mapper["models"][model_name] = model_folder_path
    save_dataset_mapper(mapper)
    return model_folder_path


class CachedDataset:
    def __init__(self, dataset_id: str, dataset_zip_url: str, datafiles_meta: dict):
        self.dataset_id = dataset_id
        self.dataset_zip_url = dataset_zip_url
        self.datafiles_meta = datafiles_meta

        check_cache_exists_or_init()

    def get_datafile_from_id(self, datafile_id):
        mapper = check_mapper_or_download(
            self.dataset_id, 
            datafile_id, 
            self.dataset_zip_url,
            self.datafiles_meta
        )
        return os.path.join(DATA_CACHE_DIR, self.dataset_id, mapper[self.dataset_id][datafile_id])



check_cache_exists_or_init()