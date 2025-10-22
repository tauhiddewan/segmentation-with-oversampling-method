import pickle
from dotenv import dotenv_values
from sklearn.model_selection import train_test_split
from utils.build import download_dataset, SubsetCreator


if __name__=="__main__":
    env_vars = dotenv_values(dotenv_path="./.env")
    download_dataset(
        url=env_vars["url"],
        data_folder_path=env_vars["data_folder_path"], 
        zfname=env_vars["dataset_zfname"],
        dataset_name=env_vars["dataset_name"],
        subset_folder_path=env_vars["subset_folder_path"], 
        output_folder_path=env_vars["output_folder_path"])
    
    subset_creator = SubsetCreator(
        dataset_path=f'{env_vars["data_folder_path"]}/{env_vars["dataset_name"]}',
        subset_folder_path = env_vars["subset_folder_path"],
        n_subsets=10, 
        images_per_subset=100
        )
    
    subset_files = subset_creator.create_subsets()
    

    data = []
    for subset_path in subset_files:
        with open(f'{subset_path}', "rb") as f:
            subset_data = pickle.load(f)
        data.extend(subset_data)
        
    train_data, test_data = train_test_split(data, 
                                            test_size=float(env_vars["test_split_size"]), 
                                            random_state=42)
    
    with open(f'{env_vars["data_folder_path"]}/{env_vars["split_fname"]}', 'wb') as f:
        pickle.dump({'train_data': train_data, 'test_data': test_data}, f)
    

