# 檔案位置: processors/pose_extractor.py
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

class PoseExtractor:
    def __init__(self):
        # 初始化 MediaPipe Pose 模型
        # static_image_mode=False: 告訴 AI 這是影片，會利用上一幀來預測下一幀 (速度較快)
        # model_complexity=2: 使用最高精度模型
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_landmarks(self, video_path):
        """
        讀取影片並回傳含有骨架數據的 DataFrame
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        data = []

        # 這裡定義我們關心的關鍵點 (排球重點部位)
        # 11:左肩, 12:右肩, 13:左肘, 14:右肘, 15:左腕, 16:右腕
        # 23:左臀, 24:右臀, 25:左膝, 26:右膝, 27:左踝, 28:右踝
        keypoints_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        keypoints_names = [
            'left_shoulder', 'right_shoulder', 
            'left_elbow', 'right_elbow', 
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip', 
            'left_knee', 'right_knee', 
            'left_ankle', 'right_ankle'
        ]

        print(f"正在處理影片: {video_path} ...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 轉成 RGB (MediaPipe 需要 RGB)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            frame_data = {'frame': frame_count}

            if results.pose_landmarks:
                # 如果有抓到人，就把座標存下來
                landmarks = results.pose_landmarks.landmark
                
                for idx, name in zip(keypoints_indices, keypoints_names):
                    lm = landmarks[idx]
                    # 存入 x, y, z, visibility (可信度)
                    frame_data[f'{name}_x'] = lm.x
                    frame_data[f'{name}_y'] = lm.y
                    frame_data[f'{name}_z'] = lm.z
                    frame_data[f'{name}_v'] = lm.visibility
            else:
                # 沒抓到人，全部填 NaN (空值)
                for name in keypoints_names:
                    frame_data[f'{name}_x'] = np.nan
                    frame_data[f'{name}_y'] = np.nan
                    frame_data[f'{name}_z'] = np.nan
                    frame_data[f'{name}_v'] = 0.0

            data.append(frame_data)
            frame_count += 1

        cap.release()
        
        # 轉成 Pandas 表格，方便後續數學處理
        df = pd.DataFrame(data)
        print(f"處理完成！共提取 {len(df)} 幀數據。")
        return df

# 如果直接執行這支程式 (測試用)
if __name__ == "__main__":
    # 這裡你可以放一個自己的測試影片路徑
    extractor = PoseExtractor()
    # 假設你有個影片叫 test.mp4
    # df = extractor.extract_landmarks("test.mp4")
    # print(df.head()) # 印出前5行看看
    print("PoseExtractor 模組準備就緒。")
