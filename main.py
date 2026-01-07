# 檔案位置: main.py
import streamlit as st
import tempfile
import cv2
import numpy as np
import pandas as pd
import os

# 引入我們寫好的模組
from processors.pose_extractor import PoseExtractor
from processors.data_cleaner import DataCleaner

# --- 設定網頁標題 ---
st.set_page_config(page_title="VolleyAI 排球分析", page_icon="🏐")
st.title("🏐 VolleyAI - 智慧排球教練")
st.write("上傳影片，AI 將自動偵測骨架並修補數據。")

# --- 輔助函式：畫骨架 ---
def draw_skeleton(frame, row, width, height):
    """
    根據 DataFrame 的某一列數據，在畫面上畫出骨架
    """
    # 定義要連線的關節 (例如：左肩連到左肘)
    connections = [
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip') # 軀幹
    ]
    
    # 畫線
    for start_part, end_part in connections:
        # 檢查數據是否存在 (有些可能沒抓到)
        if pd.isna(row[f'{start_part}_x']) or pd.isna(row[f'{end_part}_x']):
            continue
            
        # 轉換座標 (0~1) -> 像素座標 (例如 1920x1080)
        start_point = (int(row[f'{start_part}_x'] * width), int(row[f'{start_part}_y'] * height))
        end_point = (int(row[f'{end_part}_x'] * width), int(row[f'{end_part}_y'] * height))
        
        # 畫綠色的線，寬度 2
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # 畫紅色的關節點
        cv2.circle(frame, start_point, 4, (0, 0, 255), -1)
        cv2.circle(frame, end_point, 4, (0, 0, 255), -1)

    return frame

# --- 主程式邏輯 ---
uploaded_file = st.file_uploader("請上傳排球影片 (.mp4)", type=["mp4", "mov"])

if uploaded_file is not None:
    # 1. 把上傳的檔案存成暫存檔 (因為 OpenCV 需要讀取實體檔案)
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.info("影片讀取成功，開始 AI 分析... (請稍候，這需要一點時間)")

    # 2. 執行「礦工」：抓取原始骨架
    extractor = PoseExtractor()
    df_raw = extractor.extract_landmarks(video_path)
    
    # 3. 執行「煉金師」：修補數據
    cleaner = DataCleaner()
    df_clean = cleaner.process(df_raw)

    st.success(f"分析完成！共處理 {len(df_clean)} 幀。")

    # 4. 顯示數據圖表 (證明數學補償有效)
    st.subheader("📊 數據分析：手腕高度變化")
    # 比較「原始數據」跟「修補後數據」的差異
    chart_data = pd.DataFrame({
        '原始 (Raw)': df_raw['right_wrist_y'],
        '修補後 (Smoothed)': df_clean['right_wrist_y']
    })
    st.line_chart(chart_data)
    st.caption("注意看：修補後的線條（橘色）應該比原始線條（藍色）更滑順，且沒有斷裂。")

    # 5. 合成影片 (把骨架畫回去)
    st.subheader("🎬 AI 視覺化重播")
    progress_bar = st.progress(0)
    
    # 讀取原影片
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 設定輸出影片
    output_path = os.path.join(tempfile.gettempdir(), "output_skeleton.mp4")
    # mp4v 是通用的編碼格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 只有當我們有這幀的數據時才畫
        if frame_idx < len(df_clean):
            row = df_clean.iloc[frame_idx]
            frame = draw_skeleton(frame, row, width, height)
        
        out.write(frame)
        frame_idx += 1
        
        # 更新進度條
        if frame_idx % 10 == 0:
            progress_bar.progress(min(frame_idx / len(df_clean), 1.0))

    cap.release()
    out.release()
    progress_bar.progress(1.0)
    
    # 顯示影片
    st.video(output_path)
