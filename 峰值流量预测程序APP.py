import streamlit as st
import joblib
import pandas as pd

# 加载优化后的模型和选中的特征
try:
    model = joblib.load('XGB_optimized.pkl')  # 使用新训练的模型
    selected_features = joblib.load('selected_features.pkl')
except Exception as e:
    st.error(f"加载失败: {str(e)}")
    st.stop()

# 定义所有可能特征的范围（支持动态输入）
full_feature_ranges = {
    "hd": {"min": 0, "max": 100, "default": 50},
    "hw": {"min": 0, "max": 100, "default": 50},
    "hb": {"min": 0, "max": 100, "default": 50},
    "S": {"min": 0, "max": 125000000, "default": 10000},
    "Vw": {"min": 0, "max": 125000000, "default": 10000},
    "Bave": {"min": 0, "max": 250, "default": 100},
}

# 界面布局
st.title("基于多种机器学习算法和SHAP特征贡献的在线Web部署")

# 仅显示前4个重要特征的输入项
st.header("经特征筛选后得到前四个最重要的特征")
feature_values = {}
for feature in selected_features:
    props = full_feature_ranges[feature]
    value = st.number_input(
        f"{feature} ({props['min']} - {props['max']})",
        min_value=props["min"],
        max_value=props["max"],
        value=props["default"],
    )
    feature_values[feature] = value

# 预测逻辑
if st.button("预测溃口峰值流量Qp"):
    try:
        # 按 selected_features 的顺序整理输入
        input_values = [feature_values[feat] for feat in selected_features]
        input_data = pd.DataFrame([input_values], columns=selected_features)
        prediction = model.predict(input_data)[0]
        st.success(f"**溃口峰值流量Qp预测:** {prediction:.2f}")
    except Exception as e:
        st.error(f"预测失败: {str(e)}")
