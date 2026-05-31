import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import scipy.stats as stats
import scipy.signal as signal
import plotly.graph_objects as go
import os

# ================= 1. 页面纯净配置 =================
st.set_page_config(page_title="肌电动作评估系统", page_icon="⚡", layout="wide")

st.title("⚡ 表面肌电 (sEMG) 智能识别与评估系统")
st.markdown("**集成 1D-CNN 与 Random Forest 的动作分类、计数与代偿检测双保险架构**")
st.markdown("---")

# ================= 2. 核心特征引擎 & 模型骨架 =================
@st.cache_resource
def load_models_and_params():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    norm_params = np.load("saved_models/norm_params.npz")
    num_features = int(norm_params['num_features'])
    
    # ⚠️ 完美修复：安全加载极值尺子，用于兼容你的 39 维模型
    try:
        max_rms_deltoid = float(norm_params['max_rms_deltoid'])
        max_rms_biceps = float(norm_params['max_rms_biceps'])
    except KeyError:
        max_rms_deltoid, max_rms_biceps = None, None
        
    scaler = joblib.load("saved_models/scaler.pkl")
    
    class SemanticEncoder(nn.Module):
        def __init__(self, input_dim):
            super(SemanticEncoder, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, input_dim)
        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)
            
    encoder = SemanticEncoder(input_dim=num_features).to(device)
    encoder.load_state_dict(torch.load("saved_models/encoder_weights.pth", map_location=device))
    encoder.eval()
    
    knn = joblib.load("saved_models/knn_model.pkl")
    rf = joblib.load("saved_models/random_forest_model.pkl")
    
    # ⚠️ 完美修复：老老实实返回 7 个变量，绝不再报 ValueError
    return device, scaler, encoder, knn, rf, max_rms_deltoid, max_rms_biceps

def extract_39_features(ch1, ch2, fs=2000):
    def extract_single(x):
        n = len(x)
        if n < 2: return [0]*18
        dx = np.diff(x)
        threshold = 0.02 * np.std(x)
        mav = np.mean(np.abs(x))
        rmsv = np.sqrt(np.mean(x**2))
        varv = np.var(x, ddof=1) if n > 1 else 0
        wl = np.sum(np.abs(dx)) / max(1, n-1)
        sign_x = np.sign(x)
        sign_x[sign_x == 0] = 1
        zc = np.sum((np.abs(np.diff(sign_x)) > 0) & (np.abs(dx) > threshold)) / max(1, n-1)
        ssc = np.sum((dx[:-1] * dx[1:] < 0) & (np.abs(dx[:-1] - dx[1:]) > threshold)) / max(1, n-2) if n > 2 else 0
        wamp = np.sum(np.abs(dx) > threshold) / max(1, n-1)
        dasdv = np.sqrt(np.mean(dx**2))
        iemg = np.sum(np.abs(x)) / max(1, n)
        logdet = np.exp(np.mean(np.log(np.abs(x) + np.finfo(float).eps)))
        skewv = stats.skew(x)
        kurtv = stats.kurtosis(x)
        try:
            win_len = min(len(x), max(64, len(x)//2))
            f, pxx = signal.welch(x, fs, nperseg=win_len)
            tot_pwr = np.sum(pxx) + np.finfo(float).eps
            mnf = np.sum(f * pxx) / tot_pwr
            cpower = np.cumsum(pxx)
            mdf_idx = np.where(cpower >= tot_pwr / 2)[0]
            mdf = f[mdf_idx[0]] if len(mdf_idx) > 0 else 0
            pkf = f[np.argmax(pxx)]
            b_low = np.sum(pxx[(f >= 20) & (f < 60)]) / tot_pwr
            b_mid = np.sum(pxx[(f >= 60) & (f < 200)]) / tot_pwr
            b_high = np.sum(pxx[(f >= 200) & (f <= min(450, fs/2))]) / tot_pwr
        except:
            mnf, mdf, pkf, b_low, b_mid, b_high = 0,0,0,0,0,0
        return [mav, rmsv, varv, wl, zc, ssc, wamp, dasdv, iemg, logdet, skewv, kurtv, mnf, mdf, pkf, b_low, b_mid, b_high]

    f1 = extract_single(ch1)
    f2 = extract_single(ch2)
    rms_ratio = f1[1] / (f2[1] + np.finfo(float).eps)
    mav_ratio = f1[0] / (f2[0] + np.finfo(float).eps)
    corr = np.corrcoef(ch1, ch2)[0, 1] if np.std(ch1) > 0 and np.std(ch2) > 0 else 0
    return f1 + f2 + [rms_ratio, mav_ratio, float(np.nan_to_num(corr))]

# ================= 3. 侧边栏 =================
with st.sidebar:
    st.header("⚙️ 诊断控制台")
    uploaded_file = st.file_uploader("📁 上传 sEMG 数据 (.csv)", type=["csv"])
    st.markdown("---")
    st.caption("by yy")
    st.caption("累死我了")

# ================= 4. 主逻辑与视图 =================
if uploaded_file is not None:
    with st.spinner("🧠 正在进行信号提取与全局扫描..."):
        # 严格接收 7 个变量
        device, scaler, encoder, knn, rf, max_rms_deltoid, max_rms_biceps = load_models_and_params()
    
    df = pd.read_csv(uploaded_file)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    data = df.values
    fs = 2000 
    window_size, step_size = 200, 100
    action_names = {0: "弯举", 1: "推肩", 2: "侧平举"}
    
    # ---------------- 步骤一：全局预测 (一字不差复刻 predict.py) ----------------
    file_features = []
    
    for i in range(0, len(data) - window_size, step_size):
        file_features.append(extract_39_features(data[i:i+window_size, 0], data[i:i+window_size, 1], fs))
    
    if len(file_features) > 0:
        X_raw = np.array(file_features)
        
        # ⚠️ 完美修复：重新加回你原本模型必须的物理除法！恢复预测准确率！
        if max_rms_deltoid is not None:
            X_raw[:, 0] = X_raw[:, 0] / max_rms_deltoid
            X_raw[:, 1] = X_raw[:, 1] / max_rms_biceps
            
        X_scaled = scaler.transform(X_raw)
        
        tensor_X = torch.FloatTensor(X_scaled).to(device)
        with torch.no_grad():
            semantic = encoder(tensor_X).cpu().numpy()
        cnn_preds = knn.predict(semantic)
        rf_preds = rf.predict(X_scaled)
        
        total_votes = np.concatenate([cnn_preds, rf_preds])
        final_winner_idx = np.bincount(total_votes).argmax()
        final_winner = action_names[final_winner_idx]
        
        cnn_winner = action_names[np.bincount(cnn_preds).argmax()]
        rf_winner = action_names[np.bincount(rf_preds).argmax()]
    
    # ---------------- 步骤二：强力滤波与自动寻峰 (保留个数完全正确的逻辑) ----------------
    abs_sum = np.abs(data[:, 0]) + np.abs(data[:, 1])
    b, a = signal.butter(4, 1.0 / (fs / 2.0), 'low') 
    envelope = signal.filtfilt(b, a, abs_sum)
    
    peak_height_threshold = np.max(envelope) * 0.20
    peak_prominence_threshold = np.max(envelope) * 0.20
    
    peaks, _ = signal.find_peaks(
        envelope, distance=int(fs*2.5), height=peak_height_threshold, prominence=peak_prominence_threshold
    )
    total_count = len(peaks)
    
    # ---------------- 步骤三：上帝视角映射 + 20% 严苛打分 ----------------
    standard_peaks_x, standard_peaks_y = [], []
    invalid_peaks_x, invalid_peaks_y = [], []
    standard_count = 0
    
    if total_count > 0:
        st.write(f"⏳ **成功定位 {total_count} 个精华发力区，正在进行 20% 及格线判卷...**")
        progress_bar = st.progress(0)
        
        slice_radius = int((fs * 0.4) / step_size) 
        
        for idx, p in enumerate(peaks):
            slice_center_idx = p // step_size
            start_slice = max(0, slice_center_idx - slice_radius)
            end_slice = min(len(cnn_preds), slice_center_idx + slice_radius + 1)
            
            local_cnn_votes = cnn_preds[start_slice:end_slice]
            local_rf_votes = rf_preds[start_slice:end_slice]
            local_combined_votes = np.concatenate([local_cnn_votes, local_rf_votes])
            
            if len(local_combined_votes) > 0:
                # ⚠️ 完美修复：必须是 20% 的切片认同正确的全局答案，才算标准动作！
                match_ratio = np.sum(local_combined_votes == final_winner_idx) / len(local_combined_votes)
                
                if match_ratio >= 0.20:
                    standard_count += 1
                    standard_peaks_x.append(p)
                    standard_peaks_y.append(envelope[p])
                else:
                    invalid_peaks_x.append(p)
                    invalid_peaks_y.append(envelope[p])
            else:
                invalid_peaks_x.append(p)
                invalid_peaks_y.append(envelope[p])
                
            progress_bar.progress((idx + 1) / total_count)
        progress_bar.empty()

    # ---------------- 步骤四：绘制质检图谱 ----------------
    if len(file_features) > 0:
        st.subheader("📊 动作质量溯源图谱")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=data[:, 0], mode='lines', name='CH1: 三角肌', line=dict(color='#3b82f6', width=1), opacity=0.3))
        fig.add_trace(go.Scatter(y=data[:, 1], mode='lines', name='CH2: 二头肌', line=dict(color='#f97316', width=1), opacity=0.3))
        fig.add_trace(go.Scatter(y=envelope, mode='lines', name='发力包络线', line=dict(color='#374201', width=2), opacity=0.8))
        
        if len(standard_peaks_x) > 0:
            fig.add_trace(go.Scatter(x=standard_peaks_x, y=standard_peaks_y, mode='markers', 
                                     name='标准动作 ⭐', marker=dict(color='#10b981', size=18, symbol='star')))
        if len(invalid_peaks_x) > 0:
            fig.add_trace(go.Scatter(x=invalid_peaks_x, y=invalid_peaks_y, mode='markers', 
                                     name='代偿/争议动作 ❌', marker=dict(color='#ef4444', size=14, symbol='x')))
        
        fig.add_hline(y=peak_height_threshold, line_dash="dash", line_color="gray", annotation_text="波峰捕获阈值")

        fig.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
            xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"), yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---------------- 步骤五：诊断结果展示 ----------------
        st.markdown("---")
        st.subheader("🏆 全局动作种类联合诊断")
        
        c1, c2, c3 = st.columns(3)
        c1.metric(label="🧠 模型一倾向 (CNN)", value=cnn_winner)
        c2.metric(label="🌲 模型二倾向 (随机森林)", value=rf_winner)
        
        with c3:
            if cnn_winner == rf_winner:
                st.success(f"**最终判定：{final_winner}**")
            else:
                st.warning(f"**启动纠偏判定：{final_winner}**")

        st.markdown("---")
        st.subheader("📈 局部发力质量评估")
        
        c4, c5 = st.columns(2)
        c4.metric(label="🔢 捕捉到的真实波峰总数", value=f"{total_count} 次")
        
        delta_color = "normal" if standard_count == total_count else "inverse"
        delta_text = "动作全部高标准达标" if standard_count == total_count else f"含 {total_count - standard_count} 个未达 20% 的模型争议动作"
        c5.metric(label="🎯 算法判定为【标准 ⭐】的次数", value=f"{standard_count} 次", delta=delta_text, delta_color=delta_color)
        
    else:
        st.error("📉 未能提取有效特征，请检查 CSV 数据格式。")
else:
    st.info("👆 等待在左侧上传数据...")