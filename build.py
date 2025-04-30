from dotenv import dotenv_values
from utils.build import download_dataset, KvasirSubsetCreator


if __name__=="__main__":
    env_vars = dotenv_values(dotenv_path="./.env")
    download_dataset(
        url=env_vars["url"],
        data_folder_path=env_vars["data_folder_path"], 
        zfname=env_vars["dataset_zfname"],
        dataset_name=env_vars["dataset_name"],
        subset_folder_path=env_vars["subset_folder_path"], 
        output_folder_path=env_vars["output_folder_path"])
    
    subset_creator = KvasirSubsetCreator(
        dataset_path=f'{env_vars["data_folder_path"]}/{env_vars["dataset_name"]}',
        subset_folder_path = env_vars["subset_folder_path"],
        n_subsets=10, 
        images_per_subset=100
        )
    
    subset_list = subset_creator.create_subsets()
    

    for subset in subset_list:
        print(f'Subset Name : {subset}')
    

