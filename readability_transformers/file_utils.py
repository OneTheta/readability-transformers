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
import json
import requests
import shutil
import zipfile
import tarfile
import pickle


DATA_CACHE_DIR = os.path.expanduser("~/.cache/readability-transformers/data")
MAPPER_PATH = os.path.join(DATA_CACHE_DIR, "mapper.json")

def download_file(url, dest):
    local_filename = url.split('/')[-1]
    dest = os.path.join(dest, local_filename)
    with requests.get(url, stream=True) as r:
        with open(dest, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

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
        zip_ref = zipfile.ZipFile(zipped_file_path, "r")
        zip_ref.extractall(dest_url)
        zip_ref.close()

    return


def check_cache_exists_or_init():
    os.makedirs(os.path.expanduser('~/.cache/readability-transformers'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.cache/readability-transformers/data'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.cache/readability-transformers/data/all'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.cache/readability-transformers/data/preapply'), exist_ok=True)

    if not os.path.isfile(MAPPER_PATH):
        json.dump({"all": dict(), "preapply": dict()}, open(MAPPER_PATH, "w"))
    
    return True

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
