import state
import gui
from model import PointHistoryClassifier
import obswebsocket
def getSceneList():
    global scenes
    data = client.call(obswebsocket.requests.GetSceneList())
    print(data)

    scenes = data.datain['scenes']

def getSceneItemList(name):
    response = client.call(obswebsocket.requests.GetSceneItemList(sceneName=name))

    # Check if the 'sceneItems' key exists in the response
    if 'sceneItems' in response.datain:
        state.items[name] = response.datain['sceneItems']
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
        getSceneItemList(scene['sceneName'])

        # Clear existing widgets
    for widget in state.root.winfo_children():
        widget.destroy()
    state.root.update()
    gui.show_label_screen()

def sendAction():
    pass