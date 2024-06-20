"""Marnix Romeijn
[!!!INSERT DATE WHEN FINISHED!!!]
[!!!INSERT DESCRIPTION OF FUNCTION!!!]
"""

"""
When processing files, make sure they comply with the following rules:"
1. the files are in locationid_datestring_orbitnumber.nc format
2. the locationid can be a number or name for the location
3. the datestring has to be in the same format for all files
!!! add more rules if you come across anything!!!
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import time
import seaborn as sns
from sklearn import tree
import geopandas as gpd
import folium
import shap
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap
from Dashboard import SatelliteDashboard
from datetime import datetime
from collections import defaultdict
import json
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

#for laptop
# os.environ["ESMFMKFILE"] = "C:/Users/mqrom/anaconda3/envs/bsc-marnix-util-2024/Library/lib/esmf.mk"
#for pc
os.environ["ESMFMKFILE"] = "C:/Users/mqrom/miniconda3/envs/bsc-marnix-util-2024-mini/Library/lib/esmf.mk"
import xesmf as xe


def process_data(imagesdirectorypath, csvpath, saveregridspath):
    """
    Load, read, all locations within the images directory. Processes the data to comply with at least 32x32, at least
    have 20% of the methane pixels have a value. Then it finds the extreme values for latitude and longitude. After that
    every location has its captured scenes which are valid regridded with the cordinates in the csv file as the center,
    and saved in new files, one file for every location
    id in the csv file. The files are saved in the saveregridspath. After running this once, function can be commented
    out, and valid regridded data can be loaded in form the saveregridspath to save processing time.
    When processing files, make sure they comply with the following rules:
        1. the files are in locationid_datestring_orbitnumber.nc format
        2. the locationid can be a number or name for the location
        3. the datestring has to be in the same format for all files

    Parameters
    ----------
    imagesdirectorypath: directory of nc files, every file containing a single scene for a location.
    csvpath: path to csv file containing location information
    saveregridspath: path to directory where you want to save the regridded data.

    Returns Nothing, but does save data when running.
    -------

    """
    # load all nc files by iterating through imagedir, key=location; item = dict with 'datestr+orbitnr':dataarray
    rawdata = {}
    concateddata = {}

    with open(csvpath, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    testno = 0

    for file in os.listdir(imagesdirectorypath):
        fileinfo = file.split('_')
        if fileinfo[0] != None:
            # print(f'{testno}: {file}')
            # check if location already in dict
            try:
                rawdata[fileinfo[0]][fileinfo[1] + fileinfo[2]] = xr.open_dataset(
                    os.path.join(imagesdirectorypath, file))
            # if location not in dict, create key for location in dict
            except KeyError:
                rawdata[fileinfo[0]] = {
                    fileinfo[1]+fileinfo[2]: xr.open_dataset(os.path.join(imagesdirectorypath, file))}
                concateddata[fileinfo[0]] = None
            testno += 1

    print("files opened:", testno)
    print("--------------------------------------------------------------------------")

    ### preprocess data as described in Schuit et al. 2023

    reset_vars = ['ground_pixel', 'scanline']
    # crs = 'EPSG:4326'
    sum([len(rawdata[i].keys()) for i in rawdata.keys()])
    new_reset_vars = [i+'_' for i in reset_vars]
    #iterate through all locations

    regridded_dir = saveregridspath

    for location in rawdata.keys():
        concatlist = []
        regriddata = []
        max_long = -np.inf
        max_lat = -np.inf
        min_long = np.inf
        min_lat = np.inf
        max_scanline = 0
        max_groundpixel = 0
        longdif = 0
        latdif = 0
        desired_row = rows[int(location)]
        center_lat = float(desired_row['lat'])
        center_lon = float(desired_row['lon'])
        list_of_useless_keys = []

        # iterate through all captures of location
        for key in rawdata[location].keys():
            ground_pixel_shape = rawdata[location][key]['ground_pixel'].shape
            scanline_shape = rawdata[location][key]['scanline'].shape
            if (int(ground_pixel_shape[0]) or int(scanline_shape[0])) < 32:
                list_of_useless_keys.append(key)
                continue

            # reset indexes for scanline and ground_pixel, save old data in 'old_'+var in Variables
            rawdata[location][key]['old_ground_pixel'] = (
            ('old_ground_pixel',), rawdata[location][key]['ground_pixel'].values)
            rawdata[location][key]['old_scanline'] = (('old_scanline',), rawdata[location][key]['scanline'].values)

            # reset the values for ground_pixel & scanline to start at 0
            rawdata[location][key]['ground_pixel'] = (('ground_pixel',), np.arange(ground_pixel_shape[0]))
            rawdata[location][key]['scanline'] = (('scanline',), np.arange(scanline_shape[0]))

            rawdata[location][key] = rawdata[location][key].rename_vars({'latitude': 'lat', 'longitude': 'lon',
                                                                         'scanline': 'x', 'ground_pixel': 'y'})

            for variable in rawdata[location][key].data_vars:
                if variable == 'pixel_area':
                    continue
                else:
                    rawdata[location][key][variable] = (('x', 'y'), rawdata[location][key][variable].values)
            rawdata[location][key]['x'] = (('x',), rawdata[location][key]['x'].values)
            rawdata[location][key]['y'] = (('y',), rawdata[location][key]['y'].values)
            rawdata[location][key]['lat'] = (('x', 'y'), rawdata[location][key]['lat'].values)
            rawdata[location][key]['lon'] = (('x', 'y'), rawdata[location][key]['lon'].values)

            # find min and max longitude and latitude for location, and maximum size of datagrid
            item_min_long = rawdata[location][key]['lon'].min().values
            item_max_long = rawdata[location][key]['lon'].max().values
            item_min_lat = rawdata[location][key]['lat'].min().values
            item_max_lat = rawdata[location][key]['lat'].max().values
            grid_width = len(rawdata[location][key]['old_scanline'])
            grid_height = len(rawdata[location][key]['old_ground_pixel'])
            if grid_width > max_scanline:
                max_scanline = grid_width

            if grid_height > max_groundpixel:
                max_groundpixel = grid_height

            if item_max_lat > max_lat:
                max_lat = item_max_lat

            if item_min_lat < min_lat:
                min_lat = item_min_lat

            if item_min_long < min_long:
                min_long = item_min_long

            if item_max_long > max_long:
                max_long = item_max_long

        print(f'{max_long, max_lat, min_long, min_lat, max_scanline, max_groundpixel}')

        # define ds_out for regridder
        ds_out = xe.util.cf_grid_2d(min_long, max_long, (max_long - min_long) / max_scanline,
                                    min_lat, max_lat, (max_lat - min_lat) / max_groundpixel)

        # need new for loop because of finding min and max lats and lons in previous loop for every location
        for key in rawdata[location].keys():
            if key in list_of_useless_keys:
                continue
            regridder = xe.Regridder(rawdata[location][key], ds_out, method='bilinear', periodic=True,
                                     unmapped_to_nan=True)
            regridded_ds = regridder(rawdata[location][key])
            try:
                # reset values to 0 for x and y at pc
                ground_pixel_shape = regridded_ds['y_new'].shape
                scanline_shape = regridded_ds['x_new'].shape
                for var in regridded_ds.data_vars:
                    regridded_ds[var] = (('lat', 'lon'), regridded_ds[var].values)
            except:
                None

            center_lat_index = np.abs(regridded_ds['lat'] - center_lat).argmin()
            center_lon_index = np.abs(regridded_ds['lon'] - center_lon).argmin()
            # replace centering with cords from locations.csv
            lower_lat = center_lat_index + 16
            lower_lon = center_lon_index + 16
            upper_lat = center_lat_index - 16
            upper_lon = center_lon_index - 16

            # Ensure that the indices are within the valid range
            lower_lat = max(lower_lat, 0)
            lower_lon = max(lower_lon, 0)
            upper_lat = max(upper_lat, 0)
            upper_lon = max(upper_lon, 0)

            centrallats_indices = np.arange(upper_lat, lower_lat)
            centrallons_indices = np.arange(upper_lon, lower_lon)
            # Extract the subset of the dataset containing 32x32 pixels around the predetermined cords
            central_subset = regridded_ds.isel(lat=centrallats_indices, lon=centrallons_indices)

            checkarray = central_subset['methane_mixing_ratio_stripe_corrected']
            if checkarray.isnull().sum().sum() > (1 - 0.2) * checkarray.size:
                continue
            concatlist.append(central_subset)

        try:
            concateddata[location] = xr.concat(concatlist, dim='time')
            saveloc = regridded_dir+r'\location_'
            concateddata[location].to_netcdf(saveloc + desired_row['id'])
        except ValueError:
            print("For location", desired_row['id'], "no concat possible. There is no valid data for this location")


def add_classification_variable(regriddedimagesdirectory, csvpath, savefile = False, saveloc=None,
                                savename=None, simpleclassification=True, scene_concat=False):
    """
    Assign classification labels to every item in every regridded scene datasets.

    Parameters
    ----------
    regriddedimagesdirectory : str of directory which contains all regridded scene datasets in .nc format
    csvpath: str of csv file path which contains info about every artifact. if it has been changed, the process_data
    function needs to be run again! otherwise rows index in the csv and the location index will not align.
    saveloc: path where the file needs to be saved
    savename: name for file
    simpleclassification : True or False, if True it also distinguishes difference in artifacts, otherwise it only
    classifies "No plume", "plume" or "artifact".

    Returns dictionary with every loaded item in the list being a dataset full of regridded scenes, updated with
    classification label. The dictionary keys are based on the id's from the csv.
    -------

    """
    # load in data
    loaded_data = dict({})
    counter = 0
    with open(csvpath, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        counter += 1

    for file in os.listdir(regriddedimagesdirectory):
        # if file.split('_')[1][1] != '8':
        #     print("WARNING, NOT ALL FILES LOADED! CHECK AROUND LINE 220")
        #     continue
        fileinfo = file.split('_')
        loaded_data[fileinfo[1]] = xr.open_dataset(
            os.path.join(regriddedimagesdirectory, file))
        counter += 1
    total_scenes = 0
    "================================================================="

    for key in loaded_data.keys():
        loaded_data[key] = loaded_data[key].drop_vars(['old_ground_pixel', 'old_scanline', 'lat', 'lon', 'orbit',
                                                       'latitude_longitude'])

    "================================================================="
    for key in loaded_data.keys():
        total_scenes += len(loaded_data[key]['time'].values)
        print(key,'scenes:',len(loaded_data[key]['time'].values))
    print("total scenes opened:", total_scenes)
    print("--------------------------------------------------------------------------\n")

    # check if simpleclassification is True or False
    if simpleclassification == True:
        classificationdict = {
            'e': 'empty',
            'p': 'plume',
            'a': 'artefact'
        }

        for key in loaded_data.keys():
            fillvalue = classificationdict[key[0]]
            label_dataarray = xr.full_like(loaded_data[key]['time'], fill_value=fillvalue,
                                           dtype=object)
            loaded_data[key]['classification_label'] = label_dataarray
    else:
        classificationdict = {
            'e': '0',
            'p': '1'
        }
        next_index = 2
        for key in loaded_data.keys():
            for row in rows:
                if row['id'] == key:
                    if key[0] in classificationdict.keys():
                        label_dataarray = xr.full_like(loaded_data[key]['time'], fill_value=int(classificationdict[key[0]]),
                                                       dtype=str)
                        loaded_data[key]['classification_label'] = label_dataarray
                    else:
                        if row['subtype'] in classificationdict.keys():
                            label_dataarray = xr.full_like(loaded_data[key]['time'],
                                                           fill_value=int(classificationdict[row['subtype']]), dtype=str)
                            loaded_data[key]['classification_label'] = label_dataarray
                        else:
                            classificationdict[row['subtype']] = str(next_index)
                            next_index += 1
                            label_dataarray = xr.full_like(loaded_data[key]['time'],
                                                           fill_value=int(classificationdict[row['subtype']]), dtype=str)
                            loaded_data[key]['classification_label'] = label_dataarray
    print('labels')
    for key in classificationdict.keys():
        print(f'for location id {key}X label is {classificationdict[key]}')
    print("--------------------------------------------------------------------------\n")

    print('starting concat')
    near_final_data = list(loaded_data.values())

    ### use this if you want to have scenes and do not care about maintaining the timeseries
    if scene_concat:
        counter = 0
        concatlist = []
        for dataset in near_final_data:
            for timeindex in dataset['time']:
                concatlist.append(dataset.sel(time=timeindex))
                counter += 1
        final_data = xr.concat(concatlist, dim='scene', join='exact')
        savefilename = f'{os.path.join(saveloc, "_all_locations")}.nc'
        final_data.to_netcdf(savefilename)
        print(f'file saved as {savefilename}')
        print('concat done')

    # use this if you want to have multiple files which are still a timeseries
    if savefile:
        final_data = loaded_data
        for key, timeserie in final_data.items():
            savepath = f'{os.path.join(saveloc, savename)}_{key}.nc'
            timeserie.to_netcdf(savepath)
            print('file saved as ', savepath)
    print("--------------------------------------------------------------------------\n")
    return loaded_data


def print_calender(pre_classed_dict):
    """
    insert dict with locationid as key and dataset as value:
    - longest period per month since no scene
    - shows scenes over the year on a simple calendar
    """

    calenders = dict()
    for key in pre_classed_dict.keys():
        if len(pre_classed_dict[key]["time"]) < 1:
            print(f"\nNot enough scenes for location {key}!!!")
            print(f'total scenes {key}: {len(pre_classed_dict[key]["time"])}')
            continue
        print(f'\ntotal scenes {key}: {len(pre_classed_dict[key]["time"])}')
        print('Month | Scene captured for date:', key, '|max_interval between scenes')
        month_dict = {
            '01': [31*'.',0],
            '02': [28*'.'+3*' ',0],
            '03': [31*'.',0],
            '04': [30*'.'+' ',0],
            '05': [31*'.',0],
            '06': [30*'.'+' ',0],
            '07': [31*'.',0],
            '08': [31*'.',0],
            '09': [30*'.'+' ',0],
            '10': [31*'.',0],
            '11': [30*'.'+' ',0],
            '12': [31*'.',0]
        }
        for date in pre_classed_dict[key]['time'].values:
            month = str(date)[5:7]
            day = str(date)[8:10]
            if day[0] == '0':
                day = day[1]
            chars = list(month_dict[month][0])
            chars[int(day)-1] = 'X'
            month_dict[month][0] = ''.join(chars)

        counter = 0
        for key2 in month_dict.keys():
            for value in month_dict[key2][0]:
                if value == 'X':
                    month_dict[key2][1] = max(month_dict[key2][1], counter)
                    counter = 0
                    continue
                counter += 1
        month_dict[key2][1] = max(month_dict[key2][1], counter)

        calenders[key] = month_dict

        for key2 in month_dict.keys():
            print(key2, month_dict[key2])

    return calenders


def correlation_prepare_two_images(image_dataarray1, image_dataarray2):
    """
    Prepares two 2D dataarrays of the same size to contain the same amount of pixels having a value. needed to make sure
    all values are of equal weight when getting the correlation between the two dataarrays.
    Parameters
    ----------
    image_dataarray1: first data-array
    image_dataarray2: second data-array

    Returns corrected versions of both images. ready to be used to calculate the correlation between the two data-arrays
    -------

    """
    nan_mask_array1 = xr.ufuncs.isnan(image_dataarray1)
    nan_mask_array2 = xr.ufuncs.isnan(image_dataarray2)
    corrected_image1 = image_dataarray1
    corrected_image2 = image_dataarray2
    for first in range(len(nan_mask_array1)):
        for second in range(len(nan_mask_array1[first])):
            bool1 = nan_mask_array1[first][second].values
            bool2 = nan_mask_array2[first][second].values
            if bool1 != bool2:
                if bool1 == True:
                    corrected_image2[first][second] = np.nan
                if bool2 == True:
                    corrected_image1[first][second] = np.nan

    return corrected_image1, corrected_image2


def correlation_calculate_abs_corr_per_duo(classed_data, keys_of_locs, useless_correlations, remove_vars):
    dict_of_correlations = dict()
    for key in keys_of_locs:
        dict_of_correlations[key] = dict()

    for key in keys_of_locs:
        location_ds = classed_data[key]
        varlist_to_pop = list(classed_data[key].data_vars)
        varlist_to_pop.pop() # pops classification label
        if len(remove_vars) > 0:
            for i in remove_vars:
                varlist_to_pop.remove(i)
        corr_sets_checked = []
        useless_checks = useless_correlations
        for variable in varlist_to_pop:
            dict_of_correlations[key][variable] = []
            for variable2 in varlist_to_pop:
                if {variable, variable2} in useless_checks:
                    continue
                first_var_ds = classed_data[key][variable]
                second_var_ds = classed_data[key][variable2]
                if variable == variable2:
                    if {variable} in useless_checks:
                        continue
                    "when var=var the correlation will be calculated var(t=1) -> var(t=2)"
                    max_time = len(first_var_ds['time'])-1
                    timer = 0
                    while timer < max_time:
                        first_var_ds_corrected, second_var_ds_corrected = correlation_prepare_two_images(
                            first_var_ds.isel(time=timer), second_var_ds.isel(time=timer+1))
                        correlation = xr.corr(first_var_ds_corrected, second_var_ds_corrected).values
                        dict_of_correlations[key][variable].append(correlation)
                        timer += 1
                else:
                    if {variable, variable2} in corr_sets_checked:
                        continue
                    corr_sets_checked.append({variable, variable2})
                    "when different vars, calculate corr var1(t=1) -> var2(t=1)"
                    max_time = len(first_var_ds['time'])
                    timer = 0
                    while timer < max_time:
                        first_var_ds_corrected, second_var_ds_corrected = correlation_prepare_two_images(
                            first_var_ds.isel(time=timer), second_var_ds.isel(time=timer))
                        correlation = xr.corr(first_var_ds_corrected, second_var_ds_corrected).values
                        try:
                            dict_of_correlations[key][f'{variable} X {variable2}'].append(correlation)
                        except KeyError:
                            dict_of_correlations[key][f'{variable} X {variable2}'] = []
                            dict_of_correlations[key][f'{variable} X {variable2}'].append(correlation)

                        timer += 1


        print(f'{key} correlations are done')
    print("--------------------------------------------------------------------------\n")
    return dict_of_correlations




    # for location in key
        # for variable in variables
            # for variable in variables
                # if var1 and var2 in listofcorrcalcs
    pass


def date_difference(date1, date2):
    format_str = "%Y-%m-%d"
    date1_obj = datetime.strptime(date1, format_str)
    date2_obj = datetime.strptime(date2, format_str)
    difference = abs(date2_obj - date1_obj).days
    return difference


def find_consecutive_scenes(labelled_data, ids_to_check, missingdata_days_threshold, max_set_length):
    min_set_size = max_set_length
    valid_scenes_set = dict()
    for loc in ids_to_check:
        valid_scenes_set[loc] = [[],[]]
        previousdate = None
        count = 0
        list_of_dates = []
        for timeindex in range(len(labelled_data[loc]['time'].values)):
            actualtime = labelled_data[loc]['time'].isel(time=timeindex).values
            fulldate = str(labelled_data[loc]['time'].isel(time=timeindex).values)
            date = fulldate.split('T')[0]
            if previousdate is None:
                previousdate = date
                count = 1
                list_of_dates.append(actualtime)
            else:
                difference = date_difference(previousdate, date)
                if count == 1:
                    previousdate = date
                    if difference > missingdata_days_threshold:
                        for item in list_of_dates:
                            valid_scenes_set[loc][1].append(item)
                        count = 1
                        list_of_dates = [actualtime]
                    else:
                        list_of_dates.append(actualtime)
                        count += 1
                elif count > 1 and count < max_set_length-1:
                    previousdate = date
                    if difference > missingdata_days_threshold:
                        count = 1
                        if len(list_of_dates) < min_set_size:
                            for item in list_of_dates:
                                valid_scenes_set[loc][1].append(item)
                            list_of_dates = [actualtime]
                            continue
                        valid_scenes_set[loc][0].append(set(list_of_dates))
                        list_of_dates = [actualtime]
                    else:
                        count += 1
                        list_of_dates.append(actualtime)
                else:
                    if difference > missingdata_days_threshold:
                        if len(list_of_dates) >= min_set_size:
                            valid_scenes_set[loc][0].append(set(list_of_dates))
                        else:
                            for item in list_of_dates:
                                valid_scenes_set[loc][1].append(item)
                        previousdate = date
                        count = 1
                        list_of_dates = [actualtime]
                    else:
                        list_of_dates.append(actualtime)
                        valid_scenes_set[loc][0].append(set(list_of_dates))
                        previousdate = None
                        count = 0
                        list_of_dates = []
        if len(list_of_dates) >= min_set_size:
            valid_scenes_set[loc][0].append(set(list_of_dates))
        for item in list_of_dates:
            valid_scenes_set[loc][1].append(item)


    print("Valid sets found for:")
    totalset = 0
    totaldropped = 0
    for loc in valid_scenes_set.keys():
        print(f"{loc}: {len(valid_scenes_set[loc][0])} sets found", f"dropping {len(valid_scenes_set[loc][1])} items")
        totalset += len(valid_scenes_set[loc][0])
        totaldropped += len(valid_scenes_set[loc][1])
    print("Total sets found:", totalset)
    print("Total scenes dropped:", totaldropped)

    return valid_scenes_set


def drop_scenes_after_set_making(pre_labelled_datas, drop_dict, saveloc, savefile=True):
    filtered_data = {}
    for loc in drop_dict.keys():
        for timevalue in drop_dict[loc][1]:
            pre_labelled_datas[loc] = pre_labelled_datas[loc].where(pre_labelled_datas[loc]['time'] != timevalue, drop=True)
    if savefile == True:
        for key, timeserie in pre_labelled_datas.items():
            savepath = f'{os.path.join(saveloc, "after_setcheck")}_{key}.nc'
            timeserie.to_netcdf(savepath)
            print('file saved as ', savepath)
    return pre_labelled_datas


def load_dashboard_classed_data(pre_labelled_imagesdirectory, csvpath, ignore_id_list):
    # load in data
    loaded_data = dict({})
    counter = 0
    with open(csvpath, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        counter += 1

    for file in os.listdir(pre_labelled_imagesdirectory):
        if file.split('_')[2][:2] in ignore_id_list:
            print(f"{file} ignored")
            continue
        fileinfo = file.split('_')
        loaded_data[fileinfo[2][:2]] = xr.open_dataset(
            os.path.join(pre_labelled_imagesdirectory, file))
        counter += 1
    total_scenes = 0

    "================================================================="
    for key in loaded_data.keys():
        total_scenes += len(loaded_data[key]['time'].values)
        print(key, 'scenes:', len(loaded_data[key]['time'].values))
    print("total scenes opened:", total_scenes)
    print("==========================================================================\n")

    return loaded_data


def find_labelled_sets(correct_labelled_data, setsdict):
    discardcount = 0
    usecount = 0
    valid_sets = dict()
    for loc in correct_labelled_data.keys():
        valid_sets[loc] = []
        previous_sets = setsdict[loc][0]
        for sett in previous_sets:
            labels = set()
            for item in sett:
                labels.add(str(correct_labelled_data[loc]['classification_label'].sel(time=item).values))
            if len(labels) == 1:
                valid_sets[loc].append(sett)
                usecount += 1
            else:
                discardcount += 1
                continue
    print("discardcount:", discardcount)
    print("usecount:", usecount)
    for loc in valid_sets.keys():
        print(f"{loc}: {len(valid_sets[loc])}")
    return valid_sets


def create_sets_with_data_dict(correctly_classed_data, validsets):
    """

    Parameters
    ----------
    correctly_classed_data
    validsets

    Returns a dictionary with 0 till len(validsets) as keys, and a list of xarray datasets of one timestamp. Together
    the items in one list are all consecutive scenes. Contains all variables still.
    -------

    """
    sets_with_datadict = dict()
    datapoint = 0
    for loc in validsets:
        for sett in validsets[loc]:
            sets_with_datadict[datapoint] = [loc]
            for timevalue in sett:
                sets_with_datadict[datapoint].append(correctly_classed_data[loc].sel(time=timevalue))
            datapoint += 1
    return sets_with_datadict


def convert_to_correlation_dict(setswithdata, onlyabsolutes=False):
    correlations = dict()
    for datapoint in setswithdata.keys():
        datapoint_correlations = dict()
        datapoint_size = len(setswithdata[datapoint])-1
        scenecounter = 0
        for scene in setswithdata[datapoint]:
            if type(scene) == str:
                datapoint_correlations['metadata'] = {'location_id': scene}
                continue
            scenecounter += 1
            datapoint_correlations['classification_label'] = str(scene['classification_label'].values)
            methane_var_data = scene['methane_mixing_ratio_stripe_corrected']
            scene_min = np.nanmin(methane_var_data.values)
            scene_max = np.nanmax(methane_var_data.values)
            datapoint_correlations['methane_mixing_rate_max_min_difference'+str(scenecounter)] = float(scene_max - scene_min)
            datapoint_correlations['methane_mixing_rate_max'+str(scenecounter)] = float(scene_max)

            filterout_vars = ['methane_mixing_ratio_stripe_corrected', 'classification_label']
            support_vars = [var for var in scene.data_vars if var not in filterout_vars]
            support_vars.append('methane_mixing_rate_max_min_difference')
            for support_var in support_vars:
                if support_var == 'methane_mixing_rate_max' or support_var == 'methane_mixing_rate_max_min_difference':
                    continue
                support_var_data = scene[support_var]
                methane_var_corrected, support_var_corrected = correlation_prepare_two_images(
                    methane_var_data, support_var_data)
                if onlyabsolutes:
                    correlation = float(abs(xr.corr(methane_var_corrected, support_var_corrected).values))
                else:
                    correlation = float(xr.corr(methane_var_corrected, support_var_corrected).values)
                datapoint_correlations[support_var+str(scenecounter)] = correlation
        print(datapoint)
        for key, item in datapoint_correlations.items():
            print(key, '=', item)
        correlations[datapoint] = datapoint_correlations
    return correlations


def plot_confusion_matrix(ax, conf_matrix, title, labels):
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(f'Confusion Matrix {title}')


def average_confusion_matrices(confusion_matrices):
    try:
        matrices_array = np.array(confusion_matrices)
        average_matrix = np.mean(matrices_array, axis=0)
        rounded_average_matrix = np.round(average_matrix).astype(int)
        return rounded_average_matrix
    except ValueError:
        print(confusion_matrices)
        raise ValueError("Matrix shape is incorrect.")


def expand_confusion_matrix(cm, num_labels=3):
    """
    Expand a smaller confusion matrix to a 3x3 matrix by adding zeros for missing labels.

    Parameters:
    cm (np.array): The smaller confusion matrix.
    num_labels (int): The total number of labels (default is 3).

    Returns:
    np.array: The expanded 3x3 confusion matrix.
    """
    expanded_cm = np.zeros((num_labels, num_labels), dtype=int)
    present_labels = np.unique(cm)

    # Create a mapping of present labels to their indices in the smaller confusion matrix
    label_map = {label: i for i, label in enumerate(present_labels)}

    for i, actual_label in enumerate(present_labels):
        for j, predicted_label in enumerate(present_labels):
            expanded_cm[actual_label, predicted_label] = cm[i, j]


def make_models(datalist, testsize, csv_file, randomstate=101, drop_vars=[], traintesttype=0):
    """

    Parameters
    ----------
    datalist
    testsize
    randomstate
    drop_vars
    traintesttype int from 0-3. 0 is normal, 1 takes out empty from training set, 2 takes out empty from test set, 3
    takes out all empties from data, so no empty in train and test

    Returns
    -------

    """

    def convert_to_float(value):
        try:
            return float(value.replace('−', '-'))  # Replace any Unicode minus signs with ASCII minus
        except ValueError:
            return None

    starttimer = time.time()

    locationcsv = pd.read_csv(csv_file)

    dataframe = pd.DataFrame(datalist)
    columns_to_drop = [col for col in dataframe.columns if any(var in col for var in ['geolocation_flags'])]
    dataframe_filtered = dataframe.drop(columns=columns_to_drop)
    dataframe_filtered = dataframe_filtered.dropna()



    columns_to_change_diff = [col for col in dataframe.columns if
                              any(var in col for var in ['methane_mixing_rate_max_min_difference'])]
    ccolumns_to_change_max = [col for col in dataframe.columns if
                              any(var in col for var in ['methane_mixing_rate_max'])]

    for var in columns_to_change_diff:
        dataframe_filtered[var] = dataframe_filtered[var] / 100
    for var in ccolumns_to_change_max:
        dataframe_filtered[var] = dataframe_filtered[var] / 1000

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    if len(drop_vars) > 0:
        columns_to_drop = [col for col in dataframe_filtered.columns if any(var in col for var in drop_vars)]
        dataframe_filtered = dataframe_filtered.drop(columns=columns_to_drop)

    print(f'dataframelength = {len(dataframe_filtered)}')

    X = dataframe_filtered.drop(columns=['classification_label'])
    Y = dataframe_filtered['classification_label']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testsize, random_state=101)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    indices_to_keep = y_test != 1
    indices_to_keeptrain = y_train != 1

    if traintesttype == 0:
        class_names = ['artefact', 'empty', 'plume']  # Example class names
        colors = ['orange', 'green', 'purple']
        saveloc = conf_matrix_directory
        labels = [0, 1, 2]
    elif traintesttype == 1:
        class_names = ['artefact', 'plume']  # Example class names
        colors = ['orange', 'blue']
        saveloc = conf_matrix_noemptytrain_dir
        labels = [0, 1, 2]
        y_train = y_train[indices_to_keeptrain]
        X_train = X_train[indices_to_keeptrain]
    elif traintesttype == 2:
        class_names = ['artefact', 'empty', 'plume']  # Example class names
        colors = ['orange', 'green', 'purple']
        saveloc = conf_matrix_noemptytest_dir
        labels = [0, 1, 2]
        y_test = y_test[indices_to_keep]
        X_test = X_test[indices_to_keep]
    elif traintesttype == 3:
        class_names = ['artefact', 'plume']  # Example class names
        colors = ['orange', 'blue']
        saveloc = conf_matrix_noempty_dir
        labels = [0, 2]
        y_test = y_test[indices_to_keep]
        X_test = X_test[indices_to_keep]
        y_train = y_train[indices_to_keeptrain]
        X_train = X_train[indices_to_keeptrain]
    else:
        return ValueError('Invalid traintesttype')

    X_test_metadata = X_test['metadata']
    X_train_metadata = X_train['metadata']

    X_train = X_train.drop(columns=['metadata'])
    X_test = X_test.drop(columns=['metadata'])

    feature_names = X.drop(columns=['metadata']).columns


    X_test_locations = pd.DataFrame()
    X_train_locations = pd.DataFrame()
    X_test_locations['location_id'] = X_test_metadata.apply(
        lambda x: x['location_id'])  # Extract 'locationid' from each dictionary entry
    X_train_locations['location_id'] = X_train_metadata.apply(
        lambda x: x['location_id'])
    train_location_counts = X_train_locations['location_id'].value_counts()
    test_location_counts = X_test_locations['location_id'].value_counts()
    "=========================================================================="
    param_grid = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini']
    }

    # param_grid = {
    #     'max_depth': [10, 20],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1],
    #     'criterion': ['gini']
    # }

    tree_classifier = DecisionTreeClassifier(random_state=randomstate)
    grid_search = GridSearchCV(estimator=tree_classifier, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_params_tree = grid_search.best_params_
    print(best_params_tree)

    tree_classifier = DecisionTreeClassifier(**best_params_tree, random_state=randomstate)

    all_cv_scores = []
    cv_conf_matrices = []

    for repeat in range(5):
        kf = KFold(n_splits=5, shuffle=True, random_state=repeat)

        # Perform cross-validation
        cv_scores = cross_val_score(tree_classifier, X_train, y_train, cv=kf)
        all_cv_scores.extend(cv_scores)

        cross_val_y_pred = cross_val_predict(tree_classifier, X_train, y_train, cv=kf)
        print(len(cross_val_y_pred))
        print(len(y_train))
        conf_matrix = confusion_matrix(y_train, cross_val_y_pred)
        cv_conf_matrices.append(conf_matrix)

    plt.clf()
    plt.close()
    average_conf_matrix = np.mean(cv_conf_matrices, axis=0)
    rounded_conf_matrix = np.round(average_conf_matrix).astype(int)
    matrix = sns.heatmap(rounded_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels)
    matrix.set_xlabel('Predicted Labels')
    matrix.set_ylabel('True Labels')
    plt.savefig(os.path.join(saveloc, f"Confmatrix_cv_{traintesttype}"))
    plt.clf()
    plt.close()

    tree_cv_mean = np.mean(all_cv_scores)
    tree_cv_std = np.std(all_cv_scores)

    # Train the classifier
    tree_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred_tree = tree_classifier.predict(X_test)
    y_probs_tree = tree_classifier.predict_proba(X_test)

    results_df = pd.DataFrame(
        {'Actual': y_test, 'Predicted': y_pred_tree, 'location_id': [i[0] for i in X_test_locations.values]})

    # Calculate accuracy by 'location_id'
    accuracy_by_locationid = results_df.groupby('location_id').apply(
        lambda x: accuracy_score(x['Actual'], x['Predicted']))

    dt_acc_dict = accuracy_by_locationid.to_dict()

    dt_confusion_matrices = {}

    locationcsv['lat'] = locationcsv['lat'].apply(convert_to_float)
    locationcsv['lon'] = locationcsv['lon'].apply(convert_to_float)
    locationcsv['geometry'] = locationcsv.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    gdf = gpd.GeoDataFrame(locationcsv, geometry='geometry')

    gdf = gpd.GeoDataFrame(locationcsv, geometry='geometry')
    m = folium.Map(location=[20, 0], zoom_start=2)
    for location_id, group in results_df.groupby('location_id'):
        cm = confusion_matrix(group['Actual'], group['Predicted'], labels=labels)
        dt_confusion_matrices[location_id] = cm
        try:
            location_count_train = train_location_counts[location_id]
        except KeyError:
            location_count_train = 0
        try:
            location_count_test = test_location_counts[location_id]
        except KeyError:
            location_count_test = 0

        location_count_total = location_count_train+location_count_test
        location_accuracy = dt_acc_dict[location_id]
        location = gdf[gdf['id'] == location_id]
        if not location.empty:
            lat = location['lat'].values[0]
            lon = location['lon'].values[0]
            color = plt.get_cmap('RdYlGn')(location_accuracy)
            color_hex = f'#{int(color[0] * 255):02x}{int(color[1] * 255):02x}{int(color[2] * 255):02x}'

            label = f"""
                        Location ID: {location_id}<br>
                        Accuracy: {location_accuracy:.2f}<br>
                        Train Count: {location_count_train}<br>
                        Test Count: {location_count_test}<br>
                        Total Count: {location_count_total}<br>
                        Model mean CV Score: {tree_cv_mean}<br>
                        Model std CV score: {tree_cv_std}<br>
                        """

            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color=color_hex,
                fill=True,
                fill_color=color_hex,
                fill_opacity=0.7,
                popup=folium.Popup(label, max_width=300)
            ).add_to(m)

    map_file = os.path.join(saveloc, 'dt_world.html')
    m.save(map_file)


    # Evaluate performance
    accuracy_dt = accuracy_score(y_test, y_pred_tree)

    dt_confusion_matrices['all'] = confusion_matrix(y_test, y_pred_tree, labels=labels)

    # Plot the decision tree
    plt.figure(figsize=(80, 30))
    tree.plot_tree(tree_classifier, feature_names=feature_names, filled=True, fontsize=8)

    legend_labels = {class_name: color for class_name, color in zip(class_names, colors)}
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label) for label, color in legend_labels.items()]
    plt.legend(handles=handles, labels=legend_labels.keys())

    plt.savefig(os.path.join(saveloc, f'{len(drop_vars)}_{round(accuracy_dt, 3)}_decision_tree_with_legend.png'))


    print(f"it took {time.time() - starttimer} seconds to run this build")
    print("==========================================================================")

    return [(accuracy_dt, dt_acc_dict, dt_confusion_matrices, tree_cv_mean, tree_cv_std)]


def test_make_models1(datalist, drop_var, train_test_state, csv_file):
    calculationcounter = 0

    dt = 0
    dt_avg = 0
    dt_drop_vars = []
    dt_accs_dict = defaultdict(list)
    dt_matrices = defaultdict(list)

    testsizers = [0.2]
    randomstates = [101]

    for rando in randomstates:
        for testsizer in testsizers:
            for drops in drop_var:
                if train_test_state == 0:
                    accuracys = make_models(datalist, testsizer, csv_file, rando, drops, train_test_state)
                elif train_test_state == 1:
                    accuracys = make_models(datalist, testsizer, csv_file, rando, drops, train_test_state)
                elif train_test_state == 2:
                    accuracys = make_models(datalist, testsizer, csv_file, rando, drops, train_test_state)
                elif train_test_state == 3:
                    accuracys = make_models(datalist, testsizer, csv_file, rando, drops, train_test_state)
                else:
                    print("train_test_state not good")
                    raise ValueError("train_test_state is not valid")
                if accuracys[0][0] > dt:
                    dt = accuracys[0][0]
                    dt_drop_vars = drops
                dt_avg += accuracys[0][0]
                dt_accs_dict['all'].append(accuracys[0][0])
                for loc, matrix in accuracys[0][2].items():
                    dt_matrices[loc].append(matrix)
                for key, value in accuracys[0][1].items():
                    dt_accs_dict[key].append(value)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    print("best values for Decision Tree")
    print('accuracy:', dt, '\naverage accuracy',
          dt_avg, '\ndropping:', dt_drop_vars)
    print(f'Cross-Validation Scores: ')
    print(f'Mean CV Score: {accuracys[0][3]}')
    print(f'Std CV Score: {accuracys[0][4]}')
    for key, value in dt_accs_dict.items():
        print(f"accuracy for {key}: {round(np.mean(value), 3)}")
    print("==========================================================================")
    print()

    if train_test_state == 0:
        saveloc = conf_matrix_directory
        nameedit = 'with empties in data'
        labels = ['artefact', 'empty', 'plume']
    elif train_test_state == 1:
        saveloc = conf_matrix_noemptytrain_dir
        nameedit = 'without empties in train set'
        labels = ['artefact', 'empty', 'plume']
    elif train_test_state == 2:
        nameedit = 'without empties in test set'
        saveloc = conf_matrix_noemptytest_dir
        labels = ['artefact', 'empty', 'plume']
    elif train_test_state == 3:
        nameedit = 'without empties in data'
        saveloc = conf_matrix_noempty_dir
        labels = ['artefact', 'plume']
    else:
        raise ValueError("train_test_state must be 0, 1, 2 or 3")


    num_items = len(dt_matrices)
    num_cols = 4
    num_rows = 5
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(22, 18))
    axs = axs.flatten()

    for idx, (loc, matrices) in enumerate(dt_matrices.items()):
        value = round(np.mean(dt_accs_dict[loc]), 3)
        avg_matrix = average_confusion_matrices(matrices)
        title = f"dt for {loc} items"

        plot_confusion_matrix(axs[idx], avg_matrix, title, labels)

    for i in range(len(dt_matrices), len(axs)):
        fig.delaxes(axs[i])

    fig.tight_layout()
    combined_plot_path = os.path.join(saveloc, "dt_all_confusion_matrices.png")
    fig.savefig(combined_plot_path)
    plt.close()


def test_clusterings(datalist, drop_vars):
    dataframe = pd.DataFrame(datalist)
    columns_to_drop = [col for col in dataframe.columns if any(var in col for var in ['geolocation_flags'])]
    dataframe_filtered = dataframe.drop(columns=columns_to_drop)
    dataframe_filtered = dataframe_filtered.dropna()
    dataframe_filtered = dataframe_filtered.drop(columns=['metadata'])

    if len(drop_vars) > 0:
        columns_to_drop = [col for col in dataframe_filtered.columns if any(var in col for var in drop_vars)]
        dataframe_filtered = dataframe_filtered.drop(columns=columns_to_drop)

    X = dataframe_filtered.drop(columns=['classification_label'])
    y = dataframe_filtered['classification_label']

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality Reduction
    # Using UMAP
    reducer = umap.UMAP(n_components=18)
    X_umap = reducer.fit_transform(X_scaled)


    # Print the shape of the components array
    print("Shape of components array:", X_umap.shape)

    # Clustering
    kmeans = KMeans(n_clusters=12, random_state=101)  # Adjust n_clusters as needed
    clusters = kmeans.fit_predict(X_scaled)

    symbol_mapping = {
        0: 'circle',
        1: 'square',
        2: 'diamond',
    }

    # Add more mappings as needed based on the number of unique label encodings
    # Create trace for each cluster

    def create_trace(X_reduced, clusters, labels, title):
        traces = []
        for cluster in np.unique(clusters):
            cluster_mask = clusters == cluster
            trace = go.Scatter3d(
                x=X_reduced[cluster_mask, 0],
                y=X_reduced[cluster_mask, 1],
                z=X_reduced[cluster_mask, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=cluster,  # Color based on cluster
                    symbol=[symbol_mapping[label] for label in labels[cluster_mask]],  # Symbol based on label
                    colorscale='Viridis',
                    opacity=0.8
                ),
                name=f'Cluster {cluster}'
            )
            traces.append(trace)

            # Count items of each label in the cluster
            label_counts = {}
            for label in set(labels[cluster_mask]):
                label_counts[label] = sum(labels[cluster_mask] == label)
            print(f"Cluster {cluster}: {label_counts}")

        layout = go.Layout(
            title=title,
            scene=dict(
                xaxis=dict(title='Component 1'),
                yaxis=dict(title='Component 2'),
                zaxis=dict(title='Component 3')
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return traces, layout

    # Create plots for each dimensionality reduction technique

    traces_umap, layout_umap = create_trace(X_umap, clusters, y, "UMAP Clustering (3D)")
    fig_umap = go.Figure(data=traces_umap, layout=layout_umap)

    # Save plots as HTML files
    pio.write_html(fig_umap, file=f'{data_directory}\\UMAP_Clustering_3D.html', auto_open=True)



if __name__ == "__main__":
    # start time counting
    starttime = time.time()

    # define the project path, for later directory definition.
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    # [HARDCODE] define all directories before running code to load, read and save data correctly.
    images_directory_path = os.path.join(__location__, 'data\marnix_images')
    csv_path = os.path.join(__location__, 'data\locations.csv')

    # are made if they do not exist yet.
    regridded_images_path = os.path.join(__location__, r'data\regridded_images')
    pre_labelled_data_directory = os.path.join(__location__, r'data\pre_labelled_data')
    after_set_check_directory = os.path.join(__location__, r'data\after_set_check')
    correctly_labelled_images_directory = os.path.join(__location__, r'data\correct_labelled_data')
    labelled_subcats_directory = os.path.join(__location__, r'data\labelled_subcats')
    results_directory = os.path.join(__location__, r'data3\results_id')
    data_directory = os.path.join(__location__, r'data3')
    conf_matrix_directory = os.path.join(__location__, r'data3\conf_matrix')
    conf_matrix_noempty_dir = os.path.join(__location__, r'data3\conf_matrix_noempty')
    conf_matrix_noemptytest_dir = os.path.join(__location__, r'data3\conf_matrix_noemptytest')
    conf_matrix_noemptytrain_dir = os.path.join(__location__, r'data3\conf_matrix_noemptytrain')

    if not os.path.exists(regridded_images_path):
        os.makedirs(regridded_images_path)
    if not os.path.exists(pre_labelled_data_directory):
        os.makedirs(pre_labelled_data_directory)
    if not os.path.exists(after_set_check_directory):
        os.makedirs(after_set_check_directory)
    if not os.path.exists(labelled_subcats_directory):
        os.makedirs(labelled_subcats_directory)
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    if not os.path.exists(correctly_labelled_images_directory):
        os.makedirs(correctly_labelled_images_directory)
    if not os.path.exists(conf_matrix_directory):
        os.makedirs(conf_matrix_directory)
    if not os.path.exists(conf_matrix_noempty_dir):
        os.makedirs(conf_matrix_noempty_dir)
    if not os.path.exists(conf_matrix_noemptytest_dir):
        os.makedirs(conf_matrix_noemptytest_dir)
    if not os.path.exists(conf_matrix_noemptytrain_dir):
        os.makedirs(conf_matrix_noemptytrain_dir)


    # # [HARDCODE] process data, comment after first successful run. Data loaded in by add_classification_variable()
    # # process_data(images_directory_path, csv_path, regridded_images_path)
    #
    # # load in processed data and assign labels for classification. This is the base for all further computations.
    # # [HARDCODE] after first run, and after correctly labelling everything in dashboard, savefile has to be set to False
    # # [HARDCODE] if data already correctly labelled, this can be commented/skipped, however, calendar doesn't print then
    # pre_labelled_data = add_classification_variable(
    #     regridded_images_path, csv_path, False,
    #     pre_labelled_data_directory, "labelled_data", True)
    #
    # calendar = print_calender(pre_labelled_data)
    #
    # print("--------------------------------------------------------------------------")
    # print("Filtering id's (Hardcoded)")
    #
    # pre_labelled_data_keys = list(pre_labelled_data.keys())
    #
    # # [HARDCODE] state for which id's no data is provided from the csv (only for print, has no functionality (yet))
    # missing_data = ['p1', 'p2', 'p4', 'e3', 'e9']
    #
    # print("--------------------------------------------------------------------------")
    # print(f"Missing data for {len(missing_data)} items: {missing_data}")
    #
    # # # [HARDCODE] get rid of useless locations according to own judgement/rules
    # useless = ['a6', 'a4', 'a5', 'e5', 'e7', 'a8']
    # for loc in useless:
    #     pre_labelled_data_keys.remove(loc)
    #     print(loc, 'deleted')
    #
    # print("--------------------------------------------------------------------------")
    # print("Filtering for consecutive scenes only (Before correct labels)")
    #
    # # done to make it easier to filter the scenes... If they are not part of a set of consecutive scenes, why label them
    # sets_and_drop_dict = find_consecutive_scenes(pre_labelled_data, pre_labelled_data_keys, 1, 3)
    #
    # pre_classed_data = drop_scenes_after_set_making(pre_labelled_data, sets_and_drop_dict, after_set_check_directory)
    #
    #
    # print("--------------------------------------------------------------------------")
    # print("Filtering done")
    # print(f"Continuing with locations: {pre_labelled_data.keys()}")

    # # [HARDCODE] You can change the directory from the pre labelled directory to correctly labelled directory if you
    # # want to continue from where you left off. Do remember to save before you stop! Else you have to restart again! The
    # # data is also saved when you close the viewed window
    #
    # dashboard_data_dict = load_dashboard_classed_data(correctly_labelled_images_directory, csv_path, useless)
    #
    # # [HARDCODE] runs the dashboard, comment lines to skip if already correctly labelled!
    # # Or close the window when it shows
    #
    # dashboard = SatelliteDashboard(dashboard_data_dict, correctly_labelled_images_directory)
    # dashboard.run()

    # print("--------------------------------------------------------------------------\n")
    # print("Make sure you have used the Dashboard to label every scene first! "
    #       "Or another way to label with 'plume, artefact, empty...'")
    #
    # print("Continuing with labelled data if statement above is true (manual check needed)")
    #
    # correctly_classed_data = load_dashboard_classed_data(correctly_labelled_images_directory, csv_path, useless)
    #
    #
    # print("Checking for valid scenesets based on requirements of")
    # # check if sets are still valid to get label (if scenes have different label, discard)
    # valid_sets = find_labelled_sets(correctly_classed_data, sets_and_drop_dict)
    #
    # # if valid, calculate average correlation for methane -> supporting vars
    # # 1 create dictionary with valid set as key and the data for those sets as value
    # pre_correlationdict = create_sets_with_data_dict(correctly_classed_data, valid_sets)
    #
    # # 2 calculate correlation between methane and support variable(s). Hardcode define which supportvariables are used.
    # # take average of correlations as a value for the cluster and methane X support var
    # correlationdict = convert_to_correlation_dict(pre_correlationdict)
    #
    # save correlations in json so code doesn't need a rerun.
    # file_path = os.path.join(data_directory, 'correlationdict.json')
    # with open(file_path, 'w') as file:
    #     json.dump(correlationdict, file)

    # print("Correlations saved!")

    # open correlations from file if needed
    file_path = os.path.join(data_directory, 'correlationdict.json')
    with open(file_path, 'r') as file:
        correlationdict = json.load(file)
    print("Loaded correlations!")
    data = [dict for dict in correlationdict.values()]

    dataframe = pd.DataFrame(data)
    print(len(dataframe))
    print("Running Scikit-Learn classifiers:")

    print("label mapping:")
    print(f"0 - artefact")
    print(f"1 - empty")
    print(f"2 - plume")

    datavars = ['eastward_wind', 'northward_wind', 'cloud_fraction_VIIRS_SWIR_IFOV', 'qa_value', 'surface_pressure',
                'chi_square', 'aerosol_optical_thickness_SWIR', 'aerosol_optical_thickness_NIR', 'surface_albedo_SWIR',
                'surface_albedo_NIR', 'methane_mixing_rate_max_min_difference', 'methane_mixing_rate_max']

    datavarscopys = [['eastward_wind', 'northward_wind', 'chi_square', 'cloud_fraction_VIIRS_SWIR_IFOV']
    ]

    # test_clusterings(data, datavarscopys[0])

    for datavarscopy in datavarscopys:
        print("=======================================================================")
        print("------------------------------------------------------------------------")
        print("Full sets")
        test_make_models1(data, [datavarscopy], 0, csv_path)
        print("------------------------------------------------------------------------")
        # print("No empty in trainset")
        # test_make_models1(data, [datavarscopy], 1, csv_path)
        # print("------------------------------------------------------------------------")
        # print("No empty in testset")
        # test_make_models1(data, [datavarscopy], 2, csv_path)
        # print("------------------------------------------------------------------------")
        print("No empty is train and test")
        test_make_models1(data, [datavarscopy], 3, csv_path)

    # 6 refine results.
    "=========================================================================="

    # show time code took
    endtime = time.time() - starttime
    print('it took', endtime, 'seconds to run this')
