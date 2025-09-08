# %%
import requests
import dash
import numpy as np
import pandas as pd
from datetime import datetime
file_location = '/Users/Vijayraje.Jadhav/Desktop/venv/TxGearUtil-main'

# %%
df = pd.read_csv('/Users/Vijayraje.Jadhav/Downloads/Transmission_alerts_all_vehicles.csv')

df

# %%
cdf = df[df['description'].str.contains('Transmission Shift ', case=False)]
cdf

# %%
cdf.shape

# %%
cdf['timestamp_ms'] = pd.to_datetime(cdf['timestamp']).astype('int64') // 10**6

cdf

# %%
cdf.reset_index(drop=True, inplace=True)


# %%
three_days_ms = 3 * 24 * 60 * 60 * 1000
cdf['three_days_before_ms'] = cdf['timestamp_ms'] - three_days_ms
cdf

# %%
one_day_after = 1 * 24 * 60 * 60 * 1000
cdf['one_day_after_ms'] = cdf['timestamp_ms'] + one_day_after
cdf

# %%
# function to extract the required PIDs from the string LABELS
def extract_PID_data(data, protocol, LABEL):
    if protocol == 'SAE_AVG':
        if LABEL == 'ENGINE_RPM':
            PID_TAG = 'spn_190_avg'
        elif LABEL == 'FRP':
            PID_TAG = 'spn_157_avg' # 245
        elif LABEL == 'OIL_PRESSURE':
            PID_TAG = 'spn_100_avg'
        elif LABEL == 'OIL_TEMPERATURE':
            PID_TAG = 'spn_175_avg'
        elif LABEL ==  'FUEL_RATE':
            PID_TAG = 'spn_183_avg'
        elif LABEL == 'THROTTLE':
             PID_TAG = 'spn_51_avg'
    elif protocol == 'SAE':
        if LABEL ==   'ENGINE_RPM':
            PID_TAG = '190'
        elif LABEL == 'FRP':
            PID_TAG = '157'
        elif LABEL == 'OIL_PRESSURE':
            PID_TAG = '100'
        elif LABEL == 'OIL_TEMPERATURE':
            PID_TAG = '175'
        elif LABEL == 'DEF_DOSING_RATE':
            PID_TAG = '4331'
        elif LABEL ==  'FUEL_RATE':
            PID_TAG = '183'
        elif LABEL == 'ENGINE_HOUR':
            PID_TAG = '247'
        elif LABEL == 'THROTTLE':
             PID_TAG = '51'
        elif LABEL == 'GEAR_UTILIZATION':
             PID_TAG = '523'
        elif LABEL == 'WHEEL_SPEED':
             PID_TAG = '84'
        elif LABEL == 'LOAD':
             PID_TAG = '92'
        elif LABEL == 'ACCELERATION_PEDAL':
             PID_TAG = '91'

    elif protocol == 'ISO':
        if LABEL ==   'ENGINE_RPM':
            PID_TAG = '0C'
        elif LABEL == 'ACCELERATION_PEDAL':
             PID_TAG = '49'
        elif LABEL == 'LOAD':
             PID_TAG = '04'
        elif LABEL == 'GEAR_UTILIZATION':
             PID_TAG = '164'
        elif LABEL == 'WHEEL_SPEED':
             PID_TAG = '0D'

    Time_vec = []
    Val_vec = []
    for data_cnt in range(0,len(data)):
        if "pids" in data[data_cnt]:
            if len(data[data_cnt]['pids'])>0:
                for sub_pid_cnt in range(0,len(data[data_cnt]['pids'])):  #this loop
                    State = data[data_cnt]['pids'][sub_pid_cnt]
                    if PID_TAG in State:
                        #print("--------------------------------------------IN----------------------------------")
                        Time_vec.append(State[PID_TAG]['timestamp'])
                        Val_vec.append(State[PID_TAG]['value'][0])
    return Time_vec, Val_vec 

# %%
grouped = cdf.groupby('vehicle_id')
grouped

group_sizes = grouped.size()
print(group_sizes)


# %%
cdf.vehicle_id.unique()

# %%
cdf.code.value_counts()

# %%
cdf[(cdf.vehicle_id== 1254200628196933632)]

# %%
vehicle_id = '1254200628196933632'
Start_TS = 1755460660000
End_TS =  1755807251000
country_FLAG = 'US'
if country_FLAG == 'US':
    OBD_data_path = 'https://old-data-downloader.intangles-aws-us-east-1.intangles.us/download/' + str(vehicle_id) +  "/" + str(Start_TS) + "/" + str(End_TS)
    PROTOCOL = 'SAE'
elif country_FLAG == 'IN':
    OBD_data_path = 'http://data-download.intangles.com:1883/download/' + str(vehicle_id) +  "/" + str(Start_TS) + "/" + str(End_TS) 
    PROTOCOL = 'SAE'
elif country_FLAG == 'FML':
    OBD_data_path = "http://algo-internal-apis.intangles-fml-aws-ap-south-1.fml.intangles.in/download/" + str(vehicle_id) +  "/" + str(Start_TS) + "/" + str(End_TS)
r = requests.get(OBD_data_path, stream=True)
data = r.json()   

# %%
data

# %%
PROTOCOL = 'SAE'
LABEL = 'ENGINE_RPM'
RPM_Time_vec,RPM_Val_vec = extract_PID_data(data, PROTOCOL,LABEL)
print(RPM_Time_vec)
print(RPM_Val_vec)

# %%
PROTOCOL = 'SAE'
LABEL = 'GEAR_UTILIZATION'
GEAR_RATIO_Time_vec,GEAR_RATIO_Val_vec = extract_PID_data(data, PROTOCOL,LABEL)
print(GEAR_RATIO_Time_vec)
print(GEAR_RATIO_Val_vec)

# %%
PROTOCOL = 'SAE'
LABEL = 'WHEEL_SPEED'
WHEEL_SPEED_Time_vec,WHEEL_SPEED_Val_vec = extract_PID_data(data, PROTOCOL,LABEL)
print(WHEEL_SPEED_Time_vec)
print(WHEEL_SPEED_Val_vec)

# %%
PROTOCOL = 'SAE'
LABEL = 'LOAD'
LOAD_Time_vec,LOAD_Val_vec = extract_PID_data(data, PROTOCOL,LABEL)
print(LOAD_Time_vec)
print(LOAD_Val_vec)

# %%
PROTOCOL = 'SAE'
LABEL = 'ACCELERATION_PEDAL'
ACCELERATION_PEDAL_Time_vec,ACCELERATION_PEDAL_Val_vec = extract_PID_data(data, PROTOCOL,LABEL)
print(ACCELERATION_PEDAL_Time_vec)
print(ACCELERATION_PEDAL_Val_vec)

# %%
import numpy as np
import os


# Output path
output_dir = "/Users/Vijayraje.Jadhav/Desktop/venv/TxGearUtil-main/a npz array"

# Make sure directory exists
os.makedirs(output_dir, exist_ok=True)

# File name
file_name = f"original_{vehicle_id}_{Start_TS}_{End_TS}.npz"
file_path = os.path.join(output_dir, file_name)

# Save arrays into NPZ
np.savez(
    file_path,
    RPM_Time_vec=RPM_Time_vec,
    RPM_Val_vec=RPM_Val_vec,
    GEAR_RATIO_Time_vec=GEAR_RATIO_Time_vec,
    GEAR_RATIO_Val_vec=GEAR_RATIO_Val_vec,
    WHEEL_SPEED_Time_vec=WHEEL_SPEED_Time_vec,
    WHEEL_SPEED_Val_vec=WHEEL_SPEED_Val_vec,
    LOAD_Time_vec=LOAD_Time_vec,
    LOAD_Val_vec=LOAD_Val_vec,
    ACCELERATION_PEDAL_Time_vec=ACCELERATION_PEDAL_Time_vec,
    ACCELERATION_PEDAL_Val_vec=ACCELERATION_PEDAL_Val_vec
)

print(f"✅ NPZ file saved at: {file_path}")


# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import datetime
import math
import requests
import json
import tarfile
import os
import time


def DownloadFile(url,local_filename):
    #local_filename = url.split('/')[-1]
    headers = {'Intangles-User-Token': 'Mp4dO0XhqMy7Vh7b1gy-8gGbXx5m6yCgziCwRc6LA5Jdipx5Cen4Fw0fqKuNfpD6'}
    r = requests.get(url, headers=headers)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                print("WRITING...........")
                f.write(chunk)
    return 

def ExtractJason_obd(local_filename,Temp_tar_file,out_path):
    LINK_data = [json.loads(line) for line in open(local_filename, 'r')]

    if "s3_obddata_results" in LINK_data[0]['results']['data']:   #Enable to unbrake data download
        OBD_LINKS = LINK_data[0]['results']['data']['s3_obddata_results']
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        for link_cnt in range(0,len(OBD_LINKS)):
            Link = OBD_LINKS[link_cnt]
            print(Link)
            response = requests.get(Link, stream=True)
            time.sleep(0)
            print("response.....................",response)
            if response.status_code == 200:
                with open(Temp_tar_file, 'wb') as f:
                    f.write(response.raw.read())

            if(os.path.exists(Temp_tar_file)):
                # open file
                file = tarfile.open(Temp_tar_file)
                # extracting file
                file.extractall(out_path)
                file.close()
                os.remove(Temp_tar_file)
                time.sleep(0)

        os.remove(local_filename)
        time.sleep(0)

    return
 
def ExtractJason_loc(local_filename,Temp_tar_file,out_path,vehicle_id):
    LINK_data = [json.loads(line) for line in open(local_filename, 'r')]

    # if "s3_location_results" in LINK_data[0]:   #Enable to unbrake data download
    #     if not os.path.exists(out_path):
    #         os.mkdir(out_path)
    #     for link_cnt in range(0,len(LINK_data[0]['s3_location_results'])):
    #         Link = LINK_data[0]['s3_location_results'][link_cnt]
    #         response = requests.get(Link, stream=True)
    #         time.sleep(0)
    #         print("response.....................",response)
    #         if response.status_code == 200:
    #             with open(Temp_tar_file, 'wb') as f:
    #                 f.write(response.raw.read())

    # print(LINK_data[0])
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~this is the LINK_data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~",len(LINK_data) )
    
    
    
    for link_cnt in range(0,len(LINK_data)):
        if 'mongo_location_results' in LINK_data[link_cnt]:

                if not os.path.exists(out_path):
                    os.mkdir(out_path)
            
                    
                print("------------------------------- this is the length -----------------------------------", len(LINK_data[0]['mongo_location_results']))   
                
                #print(LINK_data[link_cnt]['mongo_location_results']) 
                
                # print(LINK_data[0]['mongo_location_results'][1])
                
                data = LINK_data[link_cnt]['mongo_location_results']
                #print(data)

                file_path = out_path + '_' + str(data['hist'][0]['timestamp']) + '.json'

            # os.makedirs( file_path, exist_ok=True)

                with open(file_path, "w") as json_file:
                    json.dump(data, json_file, indent=4)

                # if(os.path.exists(Temp_tar_file)):
                #     # open file
                #     file = tarfile.open(Temp_tar_file)
                #     # extracting file
                #     file.extractall(out_path)
                #     file.close()
                #     os.remove(Temp_tar_file)
                #     time.sleep(0)

                #os.remove(local_filename)
                #time.sleep(0)


    return

# Path of CSV file
# Instead of a folder path, give a file path
local_filename = f"/Users/Vijayraje.Jadhav/Desktop/venv/TxGearUtil-main/geodata/{vehicle_id}_history.json"
Temp_tar_file  = f"/Users/Vijayraje.Jadhav/Desktop/venv/TxGearUtil-main/geodata/{vehicle_id}_temp.tar"


# vehicle_id_ls = ['1308809471333498880', '1428760973522501632', '1308809529009373184']
vehicle_id_ls = ['1254200628196933632']


for vehicle_id in vehicle_id_ls:

    Start_TS = str(int(1755460660000))
    End_TS   = str(int(1755807251000))

    # URL = "https://apis.intangles.com/vehicle/" + vehicle_id +"/obd_data/"+ Start_TS + "/" + End_TS  +"?fetch_result_from_multiple_sources=true"
    # print(URL)
    # DownloadFile(URL,local_filename)
    # out_path = '/Users/harleenkaur/Documents/Shivani_Carriers_Old_OBD/' + vehicle_id + '_obd' + '/'
    # ExtractJason_obd(local_filename,Temp_tar_file,out_path)


    URL = "https://apis.intangles-aws-us-east-1.intangles.us/vehicle/" + vehicle_id + "/history/" + Start_TS + "/" + End_TS+ "?fetch_result_from_multiple_sources=true&no_dp_filter=true&&pnum=1&psize=5000000&lang=en_US"
    print(URL)
    DownloadFile(URL,local_filename)
    out_path = '/Users/Vijayraje.Jadhav/Desktop/venv/TxGearUtil-main/geodata/' + vehicle_id + '_loc' + '/'
    ExtractJason_loc(local_filename,Temp_tar_file,out_path, vehicle_id)
    

# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import datetime
import math
import requests
import json
import tarfile
import os
import time

def DownloadFile(url,local_filename):
    #local_filename = url.split('/')[-1]
    headers = {'Intangles-User-Token': 'Mp4dO0XhqMy7Vh7b1gy-8gGbXx5m6yCgziCwRc6LA5Jdipx5Cen4Fw0fqKuNfpD6'}
    r = requests.get(url, headers=headers)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                print("WRITING...........")
                f.write(chunk)
    return 

def ExtractJason_obd(local_filename,Temp_tar_file,out_path):
    LINK_data = [json.loads(line) for line in open(local_filename, 'r')]

    if "s3_obddata_results" in LINK_data[0]['results']['data']:   #Enable to unbrake data download
        OBD_LINKS = LINK_data[0]['results']['data']['s3_obddata_results']
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        for link_cnt in range(0,len(OBD_LINKS)):
            Link = OBD_LINKS[link_cnt]
            print(Link)
            response = requests.get(Link, stream=True)
            time.sleep(0)
            print("response.....................",response)
            if response.status_code == 200:
                with open(Temp_tar_file, 'wb') as f:
                    f.write(response.raw.read())

            if(os.path.exists(Temp_tar_file)):
                # open file
                file = tarfile.open(Temp_tar_file)
                # extracting file
                file.extractall(out_path)
                file.close()
                os.remove(Temp_tar_file)
                time.sleep(0)

        os.remove(local_filename)
        time.sleep(0)

    return
 
def ExtractJason_loc(local_filename,Temp_tar_file,out_path):
    LINK_data = [json.loads(line) for line in open(local_filename, 'r')]

    if "s3_location_results" in LINK_data[0]:   #Enable to unbrake data download
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        for link_cnt in range(0,len(LINK_data[0]['s3_location_results'])):
            Link = LINK_data[0]['s3_location_results'][link_cnt]
            response = requests.get(Link, stream=True)
            time.sleep(0)
            print("response.....................",response)
            if response.status_code == 200:
                with open(Temp_tar_file, 'wb') as f:
                    f.write(response.raw.read())

            if(os.path.exists(Temp_tar_file)):
                # open file
                file = tarfile.open(Temp_tar_file)
                # extracting file
                file.extractall(out_path)
                file.close()
                os.remove(Temp_tar_file)
                time.sleep(0)

        os.remove(local_filename)
        time.sleep(0)

    return

# Path of CSV file
local_filename = f"/Users/Vijayraje.Jadhav/Desktop/venv/TxGearUtil-main/geodata/temp.json"
Temp_tar_file  = f"/Users/Vijayraje.Jadhav/Desktop/venv/TxGearUtil-main/geodata/temp.tar"
veichle_ID = '1264017215108153344'

Start_TS = str(int(1728705749000))
End_TS   = str(int(1729053168000))

# URL = "https://apis.intangles.com/vehicle/" + veichle_ID +"/obd_data/"+ Start_TS + "/" + End_TS  +"?fetch_result_from_multiple_sources=true"
# print(URL)
# DownloadFile(URL,local_filename)
# out_path = 'D:/Work/Engine_Over_run/DATA/' + veichle_ID + '_obd' + '/'
# ExtractJason_obd(local_filename,Temp_tar_file,out_path)


URL = "https://apis.intangles-aws-us-east-1.intangles.us/vehicle/" + veichle_ID + "/history/" + Start_TS + "/" + End_TS+ "?fetch_result_from_multiple_sources=true&no_dp_filter=true&&pnum=1&psize=500000&lang=en_US"
print(URL)
DownloadFile(URL,local_filename)
out_path = '/Users/Vijayraje.Jadhav/Desktop/venv/TxGearUtil-main/geodata/' + veichle_ID + '_loc' + '/'
ExtractJason_loc(local_filename,Temp_tar_file,out_path)

#exit(0)
                

# %%
import pandas as pd
import json

# Path to your JSON file
file_path = '/Users/Vijayraje.Jadhav/Desktop/venv/TxGearUtil-main/geodata/1254200628196933632_loc/_1755460706000.json'

# Load JSON data (supports nested or flat)
with open(file_path, 'r') as f:
    data = json.load(f)

# If JSON is flat (list of dicts), just convert directly
# If nested (e.g., {'hist': [{...}, {...}]}) then extract as below
if isinstance(data, dict) and 'hist' in data:
    df = pd.DataFrame(data['hist'])
else:
    df = pd.DataFrame(data)

# Write DataFrame to CSV
csv_path = file_path.replace('.json', '.csv')
df.to_csv(csv_path, index=False)

print(f"CSV saved to {csv_path}")




# %%
df

# %%
df.multi_sp.iloc[0:5]

# %%
import pandas as pd
import ast

# Load the CSV
file_path = "/Users/Vijayraje.Jadhav/Desktop/venv/TxGearUtil-main/geodata/1254200628196933632_loc/_1755460706000.csv"  # change this to your actual file path
df = pd.read_csv(file_path)

# Function to expand multi_sp into multiple rows
def expand_multi_sp(row):
    base_ts = row['timestamp']
    try:
        sp_list = ast.literal_eval(row['multi_sp'])  # Convert string to list of dicts
    except Exception as e:
        return []
    
    expanded = []
    # Reverse so last 'sp' aligns with base timestamp
    for i, sp_dict in enumerate(reversed(sp_list)):
        ts = base_ts - (i * 2000)  # subtract 2 seconds (2000 ms) for each step
        expanded.append({
            "timestamp": ts,
            "sp": sp_dict["sp"]
        })
    return expanded

# Expand the dataset
expanded_rows = []
for _, row in df.iterrows():
    expanded_rows.extend(expand_multi_sp(row))

# Convert to DataFrame
expanded_df = pd.DataFrame(expanded_rows)

# # Save to CSV (optional)
# expanded_df.to_csv("expanded_speeds.csv", index=False)

print("✅ Transformation complete. Expanded data saved to expanded_speeds.csv")


# %%
expanded_df

# %%
expanded_df.describe()

# %%
expanded_df.rename(columns={'timestamp': 'time', 'sp': 'speed'}, inplace=True)
expanded_df

# %%
# Merge and prepare Gear Ratio data


df_rpm = pd.DataFrame({
    'time': RPM_Time_vec,
    'RPM_Value': RPM_Val_vec
})

df_wheel_speed = pd.DataFrame({
    'time': WHEEL_SPEED_Time_vec,
    'WHEEL_SPEED_Value': WHEEL_SPEED_Val_vec
})

df_acc_pedal = pd.DataFrame({
    'time': ACCELERATION_PEDAL_Time_vec,
    'ACCELERATION_PEDAL_Value': ACCELERATION_PEDAL_Val_vec
})

df_gear_util =  pd.DataFrame({
    'time': GEAR_RATIO_Time_vec,
    'gear_util': GEAR_RATIO_Val_vec
})
pd.set_option("display.float_format", "{:.0f}".format)
print(df_rpm.head())
print(df_wheel_speed.head())
print(df_acc_pedal.head())
print(df_gear_util.head())





# %%
df_rpm.info()

# %%
expanded_df.info()

# %%
len(df_rpm.RPM_Value)

# %%
from scipy.interpolate import interp1d
import numpy as np

# xs = np.arange(10)
# ys = 2*xs + 1

# interp_func = interp1d(xs, ys)

# newarr = interp_func(np.arange(2.1, 3, 0.1))

# print(newarr)

# for each signal:
# gps_speed: get time_vec and value_vec
# interp_func = interp1d(gps_speed_time_vec, gps_speed_value_vec)
# gps_speed_merged = interp_func(rpm_time_vec)
# gear_merged
# accelator_pedal_position_merged
# gear_ratio
# t1, t2, t3 - between t1 and t2, we have t1's value
# 4, 10, 12
# def interpolation_function(time_vec = gps_speed_time_vec, value_vec = gps_speed_value_vec, ref_time_vec = rpm_time_vec, kind='linear' or 'previous)

df_rpm['time'] = df_rpm['time'].astype('int64')
df_acc_pedal['time'] = df_acc_pedal['time'].astype('int64')
df_gear_util['time'] = df_gear_util['time'].astype('int64')
expanded_df['time'] = expanded_df['time'].astype('int64')
expanded_df['speed'] = expanded_df['speed'].astype('float64')



interp_func = interp1d(
    expanded_df.time,
    expanded_df.speed,
    kind='linear',
    fill_value="extrapolate"
)

gps_speed_merged = interp_func(df_rpm.time)



# %%
df_rpm['interpolated_gps_speed'] = gps_speed_merged
df_rpm


# %%
interp_func = interp1d(df_acc_pedal.time, df_acc_pedal.ACCELERATION_PEDAL_Value, kind='linear')

acc_pedal_merged = interp_func(df_rpm.time)

print(len(acc_pedal_merged))
df_rpm['interpolated_acc_pedal'] = acc_pedal_merged
df_rpm

# %%
from scipy.interpolate import interp1d

interp_func = interp1d(
    df_gear_util['time'], 
    df_gear_util['gear_util'],
    kind='previous',
    bounds_error=False,
    fill_value=(df_gear_util['gear_util'].iloc[0], df_gear_util['gear_util'].iloc[-1])
)

gear_util_merged = interp_func(df_rpm['time'])
df_rpm['interpolated_gear_util'] = gear_util_merged

(df_rpm.head())


# %%
df_rpm.shape

# %%
# Avoid division by zero
df_rpm = df_rpm.copy()
df_rpm['gear_ratio'] = df_rpm['RPM_Value'] / df_rpm['interpolated_gps_speed'].replace(0, np.nan)

print(df_rpm.head())


# %%
df_rpm.isnull().sum()

# %%
# Convert epoch ms to datetime
df_rpm['datetime'] = pd.to_datetime(df_rpm['time'], unit='ms')
df_rpm.head()

# %%
save_path = "/Users/Vijayraje.Jadhav/Desktop/venv/TxGearUtil-main/a npz array/merged_csv/merged_1254200628196933632_1755460660000_1755807251000.csv"

df_rpm.to_csv(save_path, index=False)
print(f"CSV saved at: {save_path}")

# %%
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import numpy as np

# Convert the vertical line time once as datetime
vertical_line_time_1 = cdf.timestamp[30]
vertical_line_time_2 = cdf.timestamp[31]
# vertical_line_time_3 = cdf.timestamp[38]
# vertical_line_time_4 = cdf.timestamp[39]
# vertical_line_time_5 = cdf.timestamp[40]
# vertical_line_time_6 = cdf.timestamp[41]
# vertical_line_time_7 = cdf.timestamp[42]
# vertical_line_time_8 = cdf.timestamp[43]
# vertical_line_time_9 = cdf.timestamp[44]
# vertical_line_time_10 = cdf.timestamp[45]


# Convert all time vectors to datetime
ACCELERATION_PEDAL_Time_vec_datetime = pd.to_datetime(ACCELERATION_PEDAL_Time_vec, unit='ms')
LOAD_Time_vec_datetime = pd.to_datetime(LOAD_Time_vec, unit='ms')
RPM_Time_vec_datetime = pd.to_datetime(RPM_Time_vec, unit='ms')
WHEEL_SPEED_Time_vec_datetime = pd.to_datetime(WHEEL_SPEED_Time_vec, unit='ms')
GEAR_RATIO_Time_vec_datetime = pd.to_datetime(GEAR_RATIO_Time_vec, unit='ms')



# Define distinct colors for each trace
colors = ["#b41fb4", "#1e0eff", "#1219d9", "#2793d6", '#9467bd', "#1ad2c6","#12e735","#c9d21a","#d2c61a","#d2881a"]

# Create dynamic title
vehicle_plate = cdf.vehicle_plate[30]
code = cdf.code[30]
description = cdf.description[30]
from_time = cdf.timestamp[30]
dynamic_title = f"From Time = {from_time},Vehicle Plate = {vehicle_plate}, Code = {code}, Description = {description}"

# Create figure with independent x-axes (no shared_xaxes)
fig = make_subplots(rows=10, cols=1,
                    subplot_titles=[
                        "Original Acceleration Pedal Position (%)",
                        "Original Engine Load (%)",
                        "Original Engine RPM",
                        "original Wheel Speed",
                        "Original Gear Utilization",
                        "Merged RPM",
                        "Merged GPS speed",
                        "Merged Acceleration Pedal",
                        "Merged Gear Utilization",
                        "Merged Gear Ratio"


                    ])

# Add traces with colors
fig.add_trace(go.Scatter(x=ACCELERATION_PEDAL_Time_vec_datetime, y=ACCELERATION_PEDAL_Val_vec,
                         mode='markers+lines', name='Acceleration Pedal', marker=dict(color=colors[0]), line=dict(color=colors[0], width=2)),
              row=1, col=1)
fig.add_trace(go.Scatter(x=LOAD_Time_vec_datetime, y=LOAD_Val_vec,
                         mode='markers+lines', name='Engine Load', marker=dict(color=colors[1]), line=dict(color=colors[1], width=2)),
              row=2, col=1)
fig.add_trace(go.Scatter(x=RPM_Time_vec_datetime, y=RPM_Val_vec,
                         mode='markers+lines', name='Engine RPM', marker=dict(color=colors[2]), line=dict(color=colors[2], width=2)),
              row=3, col=1)
fig.add_trace(go.Scatter(x=WHEEL_SPEED_Time_vec_datetime, y=WHEEL_SPEED_Val_vec,
                         mode='markers+lines', name='Wheel Speed', marker=dict(color=colors[3]), line=dict(color=colors[3], width=2)),
              row=4, col=1)
fig.add_trace(go.Scatter(x=GEAR_RATIO_Time_vec_datetime, y=GEAR_RATIO_Val_vec,
                         mode='markers+lines', name='Gear Utilization', marker=dict(color=colors[4]), line=dict(color=colors[4], width=2)),
              row=5, col=1)
fig.add_trace(go.Scatter(x=df_rpm.datetime, y=df_rpm.RPM_Value,
                         mode='lines+markers', name='Merged RPM', marker=dict(color=colors[5]), line=dict(color=colors[5], width=2)),
              row=6, col=1)
fig.add_trace(go.Scatter(x=df_rpm.datetime, y=df_rpm.interpolated_gps_speed,
                         mode='lines+markers', name='Merged GPS speed', marker=dict(color=colors[6]), line=dict(color=colors[6], width=2)),
              row=7, col=1)
fig.add_trace(go.Scatter(x=df_rpm.datetime, y=df_rpm.interpolated_acc_pedal,
                         mode='lines+markers', name='Merged Acceleration Pedal', marker=dict(color=colors[7]), line=dict(color=colors[7], width=2)),
              row=8, col=1)
fig.add_trace(go.Scatter(x=df_rpm.datetime, y=df_rpm.interpolated_gear_util,
                         mode='lines+markers', name='Merged Gear Utilization', marker=dict(color=colors[8]), line=dict(color=colors[8], width=2)),
              row=9, col=1)
fig.add_trace(go.Scatter(x=df_rpm.datetime, y=df_rpm.gear_ratio,
                         mode='lines+markers', name='Merged Gear Ratio', marker=dict(color=colors[9]), line=dict(color=colors[9], width=2)),
              row=10, col=1)


# Add vertical red thick line shapes for each subplot - FIXED VERSION
for i in range(10):
    # For the first subplot, use 'y domain', for others use 'y2 domain', 'y3 domain', etc.
    yref_str = 'y domain' if i == 0 else f'y{i + 1} domain'
    xref_str = 'x' if i == 0 else f'x{i + 1}'
    
    fig.add_shape(
        dict(
            type='line',
            x0=vertical_line_time_1,
            x1=vertical_line_time_1,
            y0=0,
            y1=1,
            yref=yref_str,  # This makes the line span the full height of each subplot
            xref=xref_str,  # Ensure it references the correct x-axis for each subplot
            line=dict(color='red', width=3, dash='dashdot'),
        )
    )

for i in range(10):
    # For the first subplot, use 'y domain', for others use 'y2 domain', 'y3 domain', etc.
    yref_str = 'y domain' if i == 0 else f'y{i + 1} domain'
    xref_str = 'x' if i == 0 else f'x{i + 1}'
    
    fig.add_shape(
        dict(
            type='line',
            x0=vertical_line_time_2,
            x1=vertical_line_time_2,
            y0=0,
            y1=1,
            yref=yref_str,  # This makes the line span the full height of each subplot
            xref=xref_str,  # Ensure it references the correct x-axis for each subplot
            line=dict(color='green', width=3, dash='dashdot'),
        )
    )

# for i in range(6):
#     # For the first subplot, use 'y domain', for others use 'y2 domain', 'y3 domain', etc.
#     yref_str = 'y domain' if i == 0 else f'y{i + 1} domain'
#     xref_str = 'x' if i == 0 else f'x{i + 1}'
    
#     fig.add_shape(
#         dict(
#             type='line',
#             x0=vertical_line_time_3,
#             x1=vertical_line_time_3,
#             y0=0,
#             y1=1,
#             yref=yref_str,  # This makes the line span the full height of each subplot
#             xref=xref_str,  # Ensure it references the correct x-axis for each subplot
#             line=dict(color='red', width=3, dash='dashdot'),
#         )
#     )

# for i in range(6):
#     # For the first subplot, use 'y domain', for others use 'y2 domain', 'y3 domain', etc.
#     yref_str = 'y domain' if i == 0 else f'y{i + 1} domain'
#     xref_str = 'x' if i == 0 else f'x{i + 1}'
    
#     fig.add_shape(
#         dict(
#             type='line',
#             x0=vertical_line_time_4,
#             x1=vertical_line_time_4,
#             y0=0,
#             y1=1,
#             yref=yref_str,  # This makes the line span the full height of each subplot
#             xref=xref_str,  # Ensure it references the correct x-axis for each subplot
#             line=dict(color='green', width=3, dash='dashdot'),
#         )
#     )

# Show x-axis tick labels for all subplots
for i in range(6):
    fig.update_xaxes(showticklabels=True, row=i + 1, col=1)

# Set x-axis title on all subplots or just first
for i in range(6):
    fig.update_xaxes(title_text="Time (UTC)", row=i + 1, col=1)

# Continue with your layout settings and save as usual
# Optimized layout for full description display
fig.update_layout(
    height=1900,  # Slightly increased height to accommodate larger title
    width=1400,   # Increased width for better readability
    title=dict(
        text=dynamic_title,
        font=dict(size=16),  # Larger font size
        x=0.5,  # Center the title
        y=0.98,  # Position near top but leave some margin
        xanchor='center',
        yanchor='top'
    ),
    margin=dict(t=120, l=60, r=60, b=60),  # Larger top margin for multi-line title
    showlegend=False
)


# Update x-axis and y-axis labels
fig.update_xaxes(title_text="Time (UTC)", tickformat='%b %d\n%Y', row=10, col=1)
fig.update_yaxes(title_text="Pedal Position (%)", row=1, col=1)
fig.update_yaxes(title_text="Engine Load (%)", row=2, col=1)
fig.update_yaxes(title_text="RPM", row=3, col=1)
fig.update_yaxes(title_text="Speed", row=4, col=1)
fig.update_yaxes(title_text="Gear Utilization", row=5, col=1)
fig.update_yaxes(title_text= "Merged RPM", row=6, col=1)
fig.update_yaxes(title_text="Merged GPS speed", row=7, col=1)
fig.update_yaxes(title_text="Merged Acceleration Pedal", row=8, col=1)
fig.update_yaxes(title_text="Merged Gear Utilization", row=9, col=1)
fig.update_yaxes(title_text="Merged Gear Ratio", row=10, col=1)

# Prepare filename with vehicle ID
filename = f"/Users/Vijayraje.Jadhav/Desktop/venv/TxGearUtil-main/a npz array/graphs/combined_vehicle_parameters_{vehicle_id}_1.html"

# Save and open the plot
plot(fig, filename=filename, auto_open=True)

# %%


# %%



