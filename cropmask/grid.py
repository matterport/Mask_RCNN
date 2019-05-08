import os
from itertools import product
import rasterio
from rasterio import windows
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool, cpu_count

def get_tiles_for_threaded_map(ds, width, height):
    """
        Returns a list of tuple where each tuple is the window and transform information for the image chip.
                
        Args:
            ds (rasterio dataset): A rasterio object read with open()
            width (int): the width of a tile/window/chip
            height (int): height of the tile/window/chip
        Returns:
            a list of tuples, where the first element of the tuple is a window and the next is the transform
    """
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    chip_list = []
    def get_win(ds, col_off, row_off, width, height, big_window):
        """Helper func to get the window and transform for a particular section of an image

        Args:
            ds (rasterio dataset): A rasterio object read with rasterio.open()
            col_off (int): the column of the window, the upper left corner
            row_off (int): the row of the window, the upper left corner
            width (int): the width of a tile/window/chip
            height (int): height of the tile/window/chip
            big_window (rasterio.windows.Window): used to deal with windows that extend beyond the source image
        Returns:
            Returns the bounds of each image chip/tile as a rasterio window object as well as the transform
            as a tuple like (rasterio.windows.Window, transform)
        """
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        return (window, transform)
    chip_list = list(map(lambda x: get_win(ds, x[0], x[1], width, height, big_window), offsets))
    return chip_list

def chunk_chips(l, CHUNK_SIZE=10):
    """
    Takes a list and returns list of lists where each sub list is a chunk to be passed to 
    the write_by window function to be written sequentially in a single thread.
    A chunk = a list of chips. a chip is a window and transform tuple
    """ 
    return [l[i:i+CHUNK_SIZE] for i in range(0, len(l), CHUNK_SIZE)]

def write_by_window(in_path, out_dir, output_name_template, chip_list):
    """Writes out a window of a larger image given a widnow and transform. 

    Args:
        ds (rasterio dataset): A rasterio object read with open()
        out_dir (str): the output directory for the image chip
        output_name_template (str): string with curly braces for naming tiles by indices for uniquiness
        meta (dict): meta data of the ds 
        window (rasterio.windows.Window): the window to read and write
        transform (rasterio transform object): the affine transformation for the window
    Returns:
        Returns the outpath of the window that has been written as a tile
    """
    # open the part we need right here
    outpaths = []
    with rasterio.open(in_path, shared=False) as ds:
        for chip in chip_list:
            window, transform = chip
            ds.meta['transform'] = transform
            ds.meta['width'], ds.meta['height'] = window.width, window.height
            outpath = os.path.join(out_dir,output_name_template.format(int(window.col_off), int(window.row_off)))
            with rasterio.open(outpath, 'w', **ds.meta, shared=False) as outds:
                outds.write(ds.read(window=window))
            outpaths.append(outpath)
    return outpaths

def map_threads(func, sequence, MAX_THREADS=10):
    """
    Set MAX_THREADS in preprocess_config.yaml
    """
    threads = min(len(sequence), MAX_THREADS)
    pool = ThreadPool(threads)
    results = pool.map(func, sequence)
    pool.close()
    pool.join()
    return results

def map_processes(func, args_list, MAX_PROCESSES):
    """
    Set MAX_PROCESSES in preprocess_config.yaml
    args_sequence is a list of lists of args
    """
    processes = min(cpu_count(), MAX_PROCESSES)
    pool = Pool(processes)
    results = pool.starmap(func, args_list)
    pool.close()
    pool.join()
    return results

def grid_images_rasterio_controlled_threads(in_path, out_dir, output_name_template='tile_{}-{}.tif', MAX_THREADS=10, CHUNK_SIZE=10, grid_size=512):
    """Combines get_tiles_for_threaded_map, map_threads, and write_by_window to write out tiles of an image

    Args:
        in_path (str): Path to a raster for which to read with raterio.open()
        out_dir (str): the output directory for the image chip
        output_name_template (str): string with curly braces for naming tiles by indices for uniquiness
        grid_size (int): length in pixels of a side of a single window/tile/chip
    Returns:
        Returns the outpaths of the tiles.
    """
    with rasterio.open(in_path, shared=False) as src:
        all_chip_list = get_tiles_for_threaded_map(src, width=grid_size, height=grid_size)
    chunk_list = chunk_chips(all_chip_list, CHUNK_SIZE) # a chunk is a list of chips 
    return list(map_threads(lambda x: write_by_window(in_path, out_dir, output_name_template, x), chunk_list, MAX_THREADS=MAX_THREADS))

def grid_images_rasterio_controlled_processes(in_path, out_dir, output_name_template='tile_{}-{}.tif', MAX_PROCESSES=4, grid_size=512):
    """Combines get_tiles_for_threaded_map, map_threads, and write_by_window to write out tiles of an image

    Args:
        in_path (str): Path to a raster for which to read with raterio.open()
        out_dir (str): the output directory for the image chip
        output_name_template (str): string with curly braces for naming tiles by indices for uniquiness
        grid_size (int): length in pixels of a side of a single window/tile/chip
    Returns:
        Returns the outpaths of the tiles.
    """
    with rasterio.open(in_path, shared=False) as src:
        all_chip_list = get_tiles_for_threaded_map(src, width=grid_size, height=grid_size)
    processes = min(cpu_count(), MAX_PROCESSES)
    chunk_list = chunk_chips(all_chip_list, len(all_chip_list)//processes) # a chunk is a list of chips, we want a chunk for each process 
    print(len(chunk_list))
    args_list = [[in_path, out_dir, output_name_template]+[chunk] for chunk in chunk_list]
    print(len(args_list))
    return list(map_processes(write_by_window, args_list, MAX_PROCESSES=MAX_PROCESSES))
