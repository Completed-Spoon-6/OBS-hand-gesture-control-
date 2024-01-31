
import tkinter as tk
from tkinter import ttk

import state
import obs_integration
import utils_functions
from camera import intialize_camera
import const

def create_gui():
    try:
        print("Initializing GUI...")

        state.root = tk.Tk()
        state.root.title("Connection GUI")


        tk.Label(state.root, text="IP:").grid(row=0, column=0)
        state.ip_entry = tk.Entry(state.root)
        state.ip_entry.grid(row=0, column=1)
        state.ip_entry.insert(0, "192.168.1.169")


        tk.Label(state.root, text="Port:").grid(row=1, column=0)
        state.port_entry = tk.Entry(state.root)
        state.port_entry.grid(row=1, column=1)
        state.port_entry.insert(0, "4455")  # Default Port

        tk.Label(state.root, text="Password:").grid(row=2, column=0)
        state.password_entry = tk.Entry(state.root, show="*")
        state.password_entry.grid(row=2, column=1)
        state.password_entry.insert(0, "M7AB9GXvWAXRPtHP")  # Default Password

        connect_button = tk.Button(state.root, text="Connect", command=obs_integration.on_connect)
        connect_button.grid(row=3, column=1)

        state.root.mainloop()
    except Exception as e:
        print(f"An error occurred: {e}")

def save_action(label, scene_name, detail_window, item, widgets):
    # Update item data from widgets
    item[const.sceneItemEnabled] = widgets[const.sceneItemEnabled].get()
    item[const.sceneName] = scene_name
    for key in item[const.sceneItemTransform]:
        widget = widgets[key]
        if isinstance(widget, tk.BooleanVar):
            item[const.sceneItemTransform][key] = widget.get()
        else:  # Assuming it's an Entry widget
            value = widget.get()
            # Convert to appropriate type if necessary
            item[const.sceneItemTransform][key] = float(value) if value.replace('.', '', 1).isdigit() else value

    # Save the updated item to the actions dictionary
    state.actions[label] = item
    detail_window.destroy()

def show_item_details(item_name, scene_name, label):
    item = next((item for item in state.items[scene_name] if item[const.sourceName] == item_name), None)
    if item is None:
        return  # Item not found

    detail_window = tk.Toplevel()
    detail_window.title(f"{label} - {scene_name} - {item_name}")

    widgets = {}

    # Show and hide scene
    tk.Label(detail_window, text="Show").grid(row=0, column=0)
    sceneItemEnabled_var = tk.BooleanVar(value=item.get(const.sceneItemEnabled, False))
    widgets[const.sceneItemEnabled] = sceneItemEnabled_var
    tk.Checkbutton(detail_window, variable=sceneItemEnabled_var).grid(row=0, column=1)

    for i, (key, value) in enumerate(item[const.sceneItemTransform].items(), start=1):
        tk.Label(detail_window, text=f"{key}:").grid(row=i, column=0)
        if isinstance(value, bool):
            var = tk.BooleanVar(value=value)
            widgets[key] = var
            tk.Checkbutton(detail_window, variable=var).grid(row=i, column=1)
        else:
            var = tk.StringVar(value=str(value))
            widgets[key] = var
            tk.Entry(detail_window, textvariable=var).grid(row=i, column=1)

    # Save and Cancel buttons
    tk.Button(detail_window, text="Save", command=lambda: save_action(label, scene_name, detail_window, item, widgets)).grid(row=len(item[const.sceneItemTransform]) + 1, column=0)
    tk.Button(detail_window, text="Cancel", command=detail_window.destroy).grid(row=len(item[const.sceneItemTransform]) + 1, column=1)

def show_label_screen():
    utils_functions.get_labels()
    state.root.title("Label List")

    for i, label in enumerate(state.keypoint_classifier_labels):
        tk.Label(state.root, text=label).grid(row=i, column=0)

        combobox_items = [f"{scene_name} - {item[const.sourceName]}" for scene_name, scene_items in state.items.items() for item in scene_items]

        combobox = ttk.Combobox(state.root, values=combobox_items)
        combobox.grid(row=i, column=1)

        # Pass the current label as a default argument to the lambda
        btn = tk.Button(state.root, text="Select values on trigger", command=lambda c=combobox, lbl=label: on_edit_button_click(c, lbl))
        btn.grid(row=i, column=2)

    start_btn = tk.Button(state.root, text="Start", command=intialize_camera)
    start_btn.grid(row=len(state.keypoint_classifier_labels), column=0)

    # List available cameras
    cameras = utils_functions.list_available_cameras()
    camera_var = tk.StringVar(state.root)
    camera_var.set(cameras[0])  # Set default value

    # Update state.camera_id when selection changes
    def on_camera_select(event):
        camera_label = camera_var.get()
        camera_id = int(camera_label.split(' ')[-1])  # Extract camera ID from label
        state.camera_id = camera_id

    camera_label = tk.Label(state.root, text="Select Camera:")
    camera_label.grid(row=len(state.keypoint_classifier_labels), column=1)

    camera_combobox = ttk.Combobox(state.root, values=cameras, textvariable=camera_var)
    camera_combobox.grid(row=len(state.keypoint_classifier_labels) + 1, column=1)
    camera_combobox.bind("<<ComboboxSelected>>", on_camera_select)

def on_edit_button_click(combobox, label):
    selected = combobox.get()
    if selected:
        selected = selected.replace("-", "")
        scene_name, item_name = selected.split(maxsplit=1)
        show_item_details(item_name, scene_name, label)
