# 檔案位置: processors/data_cleaner.py
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

class DataCleaner:
    def __init__(self, target_fps=60):
        self.target_fps = target_fps

    def process(self, df_raw):
        """
        主流程：填補空洞 -> 平滑化
        """
        if df_raw.empty:
            return df_raw

        # 1. 填補空缺 (Interpolation)
        # 使用 spline (曲線) 補值，讓動作更自然
        df_filled = df_raw.interpolate(method='cubic', limit_direction='both')
        df_filled = df_filled.fillna(method='bfill').fillna(method='ffill')

        # 2. 平滑化 (Smoothing) - 消除 AI 偵測的抖動
        df_smoothed = df_filled.copy()
        
        # 找出所有座標欄位
        columns_to_smooth = [c for c in df_smoothed.columns if c.endswith(('_x', '_y', '_z'))]
        
        for col in columns_to_smooth:
            if len(df_smoothed) > 7: # 數據夠長才做平滑
                try:
                    # window_length=7: 參考前後7個點來平滑
                    # polyorder=2: 使用二次方程式擬合
                    df_smoothed[col] = savgol_filter(df_smoothed[col], window_length=7, polyorder=2)
                except Exception:
                    pass # 如果出錯就跳過，保持原樣

        return df_smoothed

# --- 測試區 ---
if __name__ == "__main__":
    print("DataCleaner 模組準備就緒。")
