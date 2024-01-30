import state
import gui
import utils_functions
from model import PointHistoryClassifier
import obswebsocket
from obswebsocket import requests

import const
def getSceneList():
    global scenes
    data = client.call(obswebsocket.requests.GetSceneList())
    print(data)

    scenes = data.datain[const.scenes_string]

def getSceneItemList(name):
    response = client.call(obswebsocket.requests.GetSceneItemList(sceneName=name))

    # Check if the 'sceneItems' key exists in the response
    if const.sceneItems in response.datain:
        state.items[name] = response.datain[const.sceneItems]
        print(f"Items for scene '{name}': {state.items}")
    else:
        print(f"No items found for scene '{name}', or an error occurred.")

def on_connect():
    global client
    # Connection logic
    ip = state.ip_entry.get()
    port = state.port_entry.get()
    password = state.password_entry.get()

    print(f"IP: {ip}, Port: {port}, Password: {password}")
    client = obswebsocket.obsws(ip, port, password)
    client.connect()
    getSceneList()
    for scene in scenes:
        getSceneItemList(scene[const.sceneName])

        # Clear existing widgets
    for widget in state.root.winfo_children():
        widget.destroy()
    state.root.update()
    gui.show_label_screen()

def sendAction(hand_sign_text, palm, middle_finger_base):
    x,y = utils_functions.get_center_of_bounding_rect(palm, middle_finger_base)
    print(palm)
    print(middle_finger_base)

    if hand_sign_text in state.actions.keys():
        current_action = state.actions[hand_sign_text]
        scale_x = current_action[const.sceneItemTransform][const.scaleX]
        scale_y = current_action[const.sceneItemTransform][const.scaleY]
        response1 = client.call(requests.GetSceneItemTransform( sceneName=current_action[const.sceneName], sceneItemId=current_action[const.sceneItemId]))
        print(response1)
        response = client.call(requests.SetSceneItemTransform(sceneItemTransform={const.positionX: x , const.positionY: y }, sceneName=current_action[const.sceneName], sceneItemId=current_action[const.sceneItemId]))
        print(response)
        client.call(requests.SetSceneItemEnabled(sceneItemId=current_action[const.sceneItemId], sceneName=current_action[const.sceneName], sceneItemEnabled=current_action[const.sceneItemEnabled]))
