
import socket
import struct
import time
import numpy as np
import json
from PIL import Image
import io
import cv2

class UDPSender:

    CHUNK_SIZE = 1200  # 1.2KBずつ送信（UDPのMTUを考慮）
    FRAME = 1
    INIT_PARAMS = 2
    EXTRINSIC = 3

    def __init__(self, target_ip, target_port):
        """
        コンストラクタ
        
        Parameters
            target_ip (str): 受信側のIPアドレス
            target_port (int): 受信側と合わせる
        """
        self.target_ip = target_ip
        self.target_port = target_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


    def to_bytes(self, data, quality = 0) -> bytes:
        """
        任意のデータを bytes 型に変換する関数

        Parameters
            data (any): 色々な形式のデータ
            quality (int): JPEG圧縮品質（1-100）
                            ※ 画像の場合は必ず指定
        
        Return
            data (bytes): 変換後のデータ
        """
        # すでに bytes 型ならそのまま返す
        if isinstance(data, bytes):
            return data
        
        # NumPy配列 → バイナリデータ
        # NumPy配列（OpenCV画像など） → JPEG形式でバイト列化
        if isinstance(data, np.ndarray):
            if quality == 0:
               return data.tobytes()
            else:
                # JPEGエンコード用オプション
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                success, encoded_img = cv2.imencode(".jpg", data, encode_param)
                if not success:
                    raise ValueError("画像のJPEGエンコードに失敗しました")
                return encoded_img.tobytes()
        
        # 数値や真偽値 → 文字列化してエンコード
        if isinstance(data, (int, float, bool)):
            return str(data).encode('utf-8')
        
        # 文字列 → UTF-8エンコード
        if isinstance(data, str):
            return data.encode('utf-8')
        
        # リスト・辞書など → JSON文字列にしてエンコード
        if isinstance(data, (dict, list, tuple)):
            return json.dumps(data, ensure_ascii=False).encode('utf-8')
        
        # PIL Image → PNG形式でバイト列に変換
        if isinstance(data, Image.Image):
            buf = io.BytesIO()
            data.save(buf, format='PNG')
            return buf.getvalue()
        
        raise ValueError("エンコードに失敗しました")

    
    def send(self, data):
        """
        UDP送信する関数

        Parameters
            data (bytes): 送信データ
        """        
        if not isinstance(data, bytes):
            raise TypeError("データは bytes である必要があります")
        self.sock.sendto(data, (self.target_ip, self.target_port))


    def send_frame(self, frame_data, frame_id):
        """
        映像をUDP送信する関数

        Parameters
            frame_data (bytes): 送信データ
            frame_id (int): フレームID
        """
        # パケット数（data を CHUNK_SIZEで分割した数）を先に送信
        num_packets = (len(frame_data) + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE

        # バイナリヘッダを作成して送信（format: uint32, frame_id: uint32, num_packets: uint32）- big-endian
        # 12byte = format(4byte) + frame_id(4byte) + num_packets(4byte)
        header = struct.pack("!III", self.FRAME, frame_id & 0xFFFFFFFF, num_packets)
        self.sock.sendto(header, (self.target_ip, self.target_port))

        # チャンク送信（各チャンクにも format と frame_id と seq を付加）
        for i in range(num_packets):
            chunk = frame_data[i * self.CHUNK_SIZE : (i + 1) * self.CHUNK_SIZE]
            
            # pkt → format(4B) + frame_id(4B) + seq(4B) + chunk
            pkt = struct.pack("!III", self.FRAME, frame_id & 0xFFFFFFFF, i) + chunk
            self.sock.sendto(pkt, (self.target_ip, self.target_port))
    

    def send_init_params(self, cam_shift, vfov):
        """
        カメラ初期パラメータをUDP送信する関数
        
        Parameters
            cam_shift (np.ndarray): カメラ座標軸のシフト行列 (3x3)
            vfov (float): 垂直視野角 (度)
        """
        # UDP送信用にfloat32に変換
        data = np.hstack((cam_shift.flatten(), vfov)).astype(np.float32)
        # 44byte = format(4byte) + data(4byte * 10)
        pkt = struct.pack("!I10f", self.INIT_PARAMS, *data)
        # 確実に届くように5回送信
        for _ in range(5):
            self.sock.sendto(pkt, (self.target_ip, self.target_port))
            time.sleep(0.1)


    def send_extrinsic_parameters(self, R_cam, t_cam):
        """
        カメラの外部パラメータをUDP送信する関数

        Parameters
            R_cam (np.ndarray): 回転行列 (3x3)
            t_cam (np.ndarray): 並進ベクトル (3,)
        """
        # UDP送信用にfloat32に変換
        data = np.hstack((R_cam.flatten(), t_cam.flatten())).astype(np.float32)
        # 52byte = format(4byte) + data(4byte * 12)
        pkt = struct.pack("!I12f", self.EXTRINSIC, *data)
        self.sock.sendto(pkt, (self.target_ip, self.target_port))


    def close(self):
        """ソケットを閉じる"""
        self.sock.close()