import pyzed.sl as sl
import cv2
import datetime
import copy

class ZEDCamera:

    INPUT_CAMERA = 0
    OUTPUT_FILE = 0

    #
    # コンストラクタ
    #
    def __init__(self, width, height):
        self.inputMode = self.INPUT_CAMERA
        self.outputMode = self.OUTPUT_FILE
        self.vflip = False
        self.hflip = False
        self.save_original = True

        self.OpenCamera(width, height)

        # 出力ファイル用のヘッダー生成
        date = datetime.datetime.now()
        self.output_header = "%04d-%02d-%02d-%02d:%02d:%02d" % (
            date.year,
            date.month,
            date.day,
            date.hour,
            date.minute,
            date.second,
        )

        # ビデオ出力フラグ(True: ビデオ出力中, False: ビデオ出力停止中)
        self.video_out = False

    def __del__(self):
        self.Close()

    #
    # カメラをクローズする関数
    #
    def Close(self):
        self.zed.close()

    #
    # カメラをオープンする関数
    #
    # @param width  : 画像の横サイズ
    # @param height : 画像の縦サイズ
    #
    def OpenCamera(self, width, height):
        self.inoutMode = self.INPUT_CAMERA

        # ZEDカメラの初期化
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 60
        init_params.depth_mode = sl.DEPTH_MODE.NONE # 深度不要な場合
        status = self.zed.open(init_params)

        if status != sl.ERROR_CODE.SUCCESS:
            print("Camera open error:", status)
            return False

        # カメラの視野角を取得
        cam_info = self.zed.get_camera_information()
        cam_config = cam_info.camera_configuration
        left_cam_params = cam_config.calibration_parameters.left_cam
        self.vfov = left_cam_params.v_fov
        self.hfov = left_cam_params.h_fov
        self.focus = left_cam_params.fx

        # 画像取得のためのオブジェクト
        self.image_obj = sl.Mat()

        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_obj, sl.VIEW.LEFT)   # 左目画像を取得
            image = self.image_obj.get_data()                       # numpy配列に変換（BGR）
            image = cv2.resize(image, (width, height))              # リサイズ

        self.width = width
        self.height = height
        self.nchannels = 3

        return True

    #
    # カメラまたはビデオから画像を取得する関数
    #
    def CaptureImage(self):
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_obj, sl.VIEW.LEFT)   # 左目画像を取得
            image = self.image_obj.get_data()                       # numpy配列に変換（BGR）
            image = cv2.resize(image, (self.width, self.height))    # リサイズ
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        return True, image
    
    #
    # 画像のフリップ設定を行う関数
    #
    def SetFlip(self, hflip, vflip):
        self.hflip = hflip
        self.vflip = vflip
