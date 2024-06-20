import csv
import xarray as xr

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import os

class SatelliteDashboard:
    def __init__(self, data_dict, saveloc):
        self.data_dict = data_dict
        self.saveloc = saveloc

        self.root = tk.Tk()
        self.root.title('Satellite Data Analysis')

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill='both', expand=True)

        self.frames = []
        for _ in range(3):
            frame_ = ttk.Frame(self.main_frame)
            frame_.pack(side='left', fill='both', expand=True)
            self.frames.append(frame_)

            # Add buttons to each frame
            plume_button = ttk.Button(frame_, text="Plume", command=lambda frame=frame_: self.set_data_label(frame, 'plume'))
            plume_button.pack(side='top', padx=10, pady=5)

            artefact_button = ttk.Button(frame_, text="Artefact", command=lambda frame=frame_: self.set_data_label(frame, 'artefact'))
            artefact_button.pack(side='top', padx=10, pady=5)

            empty_button = ttk.Button(frame_, text="Empty", command=lambda frame=frame_: self.set_data_label(frame, 'empty'))
            empty_button.pack(side='top', padx=10, pady=5)

        save_button = ttk.Button(self.main_frame, text="Save Data", command=self.save_data)
        save_button.pack(side='bottom', padx=10, pady=10)

        self.location_dropdowns = []
        self.variable_dropdowns = []
        self.time_sliders = []
        self.canvases = []

        for frame in self.frames:
            location_dropdown = ttk.Combobox(frame, values=list(data_dict.keys()))
            location_dropdown.pack(side='top', padx=10, pady=5)
            location_dropdown.bind("<<ComboboxSelected>>", self.location_selected)
            self.location_dropdowns.append(location_dropdown)

            variable_dropdown = ttk.Combobox(frame, values=[])
            variable_dropdown.pack(side='top', padx=10, pady=5)
            self.variable_dropdowns.append(variable_dropdown)

            time_slider = ttk.Scale(frame, orient='horizontal', length=400)  # Adjust length here
            time_slider.pack(side='top', padx=10, pady=5)
            time_slider.bind("<ButtonRelease-1>", self.update_plot)
            self.time_sliders.append(time_slider)

            canvas = FigureCanvasTkAgg(plt.figure(), master=frame)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.canvases.append(canvas)

        # Preselect the first item in dropdown menus
        for location_dropdown, variable_dropdown in zip(self.location_dropdowns, self.variable_dropdowns):
            location_dropdown.set(list(data_dict.keys())[0])
            variable_dropdown.set(list(data_dict[list(data_dict.keys())[0]].keys())[0])

    def location_selected(self, event):
        location_dropdown = event.widget
        location_id = location_dropdown.get()
        index = self.location_dropdowns.index(location_dropdown)
        variables = list(self.data_dict[location_id].keys())
        self.variable_dropdowns[index]['values'] = variables
        self.update_slider_range(index, variables)
        self.update_plot()

    def update_slider_range(self, index, variables):
        location_id = self.location_dropdowns[index].get()
        max_time = max(len(self.data_dict[location_id][variable]) for variable in variables)
        self.time_sliders[index].config(to=max_time - 1)

    def update_plot(self, event=None):
        for i, (location_dropdown, variable_dropdown, time_slider, canvas) in enumerate(
                zip(self.location_dropdowns, self.variable_dropdowns, self.time_sliders, self.canvases)):
            location_id = location_dropdown.get()
            variable = variable_dropdown.get()
            time_index = int(round(time_slider.get()))  # Round to nearest integer

            sensor_data = self.data_dict[location_id][variable][time_index]

            time_value_display = str(self.data_dict[location_id]['time'][time_index])[36:46]

            plt.figure(canvas.figure.number)
            plt.clf()  # Clear existing plot
            # plt.imshow(normalized_data, cmap='viridis', origin='lower', vmin=0, vmax=1)  # normalized
            plt.imshow(sensor_data, cmap='viridis', origin='lower') # actual data
            plt.title(f'{variable} at time {time_value_display}')
            plt.colorbar(label='Value')  # Add colorbar legend
            plt.axis('off')
            canvas.draw()

    def set_data_label(self, frame, label):
        location_dropdown = self.location_dropdowns[self.frames.index(frame)]
        location_id = location_dropdown.get()
        print(location_id)
        time_slider = self.time_sliders[self.frames.index(frame)]
        time_index = int(round((time_slider.get())))
        time = self.data_dict[location_id]['time'].isel(time=time_index).values
        print(time)

        # Set the label in the data dictionary
        print(location_id, 'label before:', self.data_dict[location_id]['classification_label'].isel(
            time=time_index).values)
        self.data_dict[location_id]['classification_label'][{'time': time_index}] = label
        print(location_id, 'changed to:', label)
        print()

    def save_data(self):
        for key, timeserie in self.data_dict.items():
            savepath = f'{os.path.join(self.saveloc, "labelled_data")}_{key}.nc'
            timeserie.to_netcdf(savepath)
            print("labels:", timeserie['classification_label'].values)
            print('file saved as ', savepath)

    def run(self):
        print("Dashboard Running")
        self.root.mainloop()
        print("Dashboard terminated")
        self.save_data()


def load_dashboard_data(labelled_imagesdirectory, csvpath, ignore_id_list):
    # load in data
    loaded_data = dict({})
    counter = 0
    with open(csvpath, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        counter += 1

    for file in os.listdir(labelled_imagesdirectory):
        if file.split('_')[2][:2] in ignore_id_list:
            print(f"{file} ignored")
            continue
        print(file.split('_')[2][:2])
        fileinfo = file.split('_')
        loaded_data[fileinfo[2][:2]] = xr.open_dataset(
            os.path.join(labelled_imagesdirectory, file))
        counter += 1
    total_scenes = 0

    "================================================================="
    for key in loaded_data.keys():
        total_scenes += len(loaded_data[key]['time'].values)
        print(key, 'scenes:', len(loaded_data[key]['time'].values))
    print("total scenes opened:", total_scenes)
    print("==========================================================================\n")

    return loaded_data

