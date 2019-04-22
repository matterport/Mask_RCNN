from azure.storage.file.fileservice import FileService
import yaml
import os


def make_regional_directories(config_path):
    """
    Sets up the directories on Azure File Storage for a 
    particular region. Requires a config file parsed from 
    yaml.
    
    Args:
        config_path (str): Path to a parsed config that contains
        the name of the region. Can be a state, country, 
        watershed, etc. but be specific to some well defined 
        boundary.
    Returns:
        Nothing, but creates the directory structure on 
        azure
    """

    with open(config_path) as f:
        configs = yaml.safe_load(f)

    file_service = FileService(
        configs["storage"]["storage_name"], configs["storage"]["storage_key"]
    )

    region_path = os.path.join(
        configs["storage"]["region_dir"], configs["storage"]["region_name"]
    )
    # anything downloaded here should be moved and processed later to the l5, etc folders on file store
    landsat_dwnld_path = os.path.join(region_path, "landsat_downloaded")
    # these folders will contain tiled Landsat imagery
    landsat_path = os.path.join(region_path, "landsat_processed")
    # rgb pngs of landsat for labeling, pre stretched
    landsat_pngs_path = os.path.join(region_path, "landsat_pngs")
    # should there be a folder for vott projects? or just one per region? or one per image source per region?

    file_service.create_directory(configs["storage"]["share_name"], region_path)
    file_service.create_directory(configs["storage"]["share_name"], landsat_path)
    file_service.create_directory(configs["storage"]["share_name"], landsat_pngs_path)

    print("done creating region directory")
