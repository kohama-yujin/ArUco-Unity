import sys
import numpy as np
import cv2
import glfw
from mqoloader.loadmqo import LoadMQO
import Application

# アプリケーションで使用するパラメータ
width = 1280
height = 720
use_api = cv2.CAP_DSHOW             # Windowsで使用する場合こちらを使う
# use_api = 0                       # Linuxで使用する場合はこちらを使う
use_zed = True                      # ZEDカメラを使用するかどうか
# udp_info = ("133.15.35.65", 12345)  # C3-502 5GHz(ip, port)
udp_info = ("127.0.0.1", 12345)     # ローカル(ip, port)
marker_size = 9.3                   # マーカーのサイズ(cm)

# アプリケーション設定
app = Application.Application("Aruco marker AR", width, height, 0, use_api, use_zed, udp_info)

# マーカーの四隅の3次元座標（+X, +Y, +Z : Roll, Pitch, Yaw)
point_3D = np.array(
    [
        (-marker_size / 2, marker_size / 2, 0.0),
        (marker_size / 2, marker_size / 2, 0.0),
        (marker_size / 2, -marker_size / 2, 0.0),
        (-marker_size / 2, -marker_size / 2, 0.0),
    ]
)
app.estimator.set_3D_points(point_3D)

# 3次元モデルの設定
model_filename = "./mqo/open_red_box.mqo"
model_scale = 1
app.use_normal = False
model = LoadMQO(model_filename, model_scale, app.use_normal)
app.set_mqo_model(model)

frame_id = 0  # フレームID初期化

# カメラの初期データを送信
app.init_udp_sender()
# アプリケーションのメインループ
while not app.glwindow.window_should_close():
    app.display_func(app.glwindow.window)

    # 以下、ボックスの開閉
    if model.meshes[0].vertices[8].z <= -20 and model.meshes[0].vertices[9].z <= -20:
        close = True
    if model.meshes[0].vertices[8].z >= -15 and model.meshes[0].vertices[9].z >= -15:
        close = False
    if close:
        model.meshes[0].vertices[8].z += 0.1
        model.meshes[0].vertices[9].z += 0.1
    else:
        model.meshes[0].vertices[8].z -= 0.1
        model.meshes[0].vertices[9].z -= 0.1

    app.udp_sender(frame_id)
    frame_id = (frame_id + 1) & 0xFFFFFFFF  # フレームIDをインクリメント（32bitでロールオーバー）

    glfw.poll_events()

glfw.terminate()
