import os
import shutil

from .file_utils import DATA_CACHE_DIR, MODEL_CACHE_DIR, MAPPER_PATH, check_cache_exists_or_init

def reset_cache():
    if os.path.isdir(DATA_CACHE_DIR):
        shutil.rmtree(DATA_CACHE_DIR)
        print("Removed:", DATA_CACHE_DIR)
    # if os.path.isdir(MODEL_CACHE_DIR):
    #     shutil.rmtree(MODEL_CACHE_DIR)
    #     print("Removed:", MODEL_CACHE_DIR)
    if os.path.isfile(MAPPER_PATH):
        os.remove(MAPPER_PATH)
        print("Removed:", MAPPER_PATH)
    check_cache_exists_or_init()
    
if __name__ == "__main__":
    reset_cache()