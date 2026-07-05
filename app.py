import streamlit as st
import pandas as pd
import numpy as np
import plotly as pl
import plotly.graph_objects as go
import random
from datetime import datetime, timedelta

# ============================
# 1. 页面基本配置
# ============================
st.set_page_config(page_title="微生物QC数据处理站", layout="wide")
st.title("🧬 微生物 QC 数据处理工具（符合 GMP 合规）")

# ============================
# 2. 全局配置：按级别的警戒限/行动限
# ============================
QC_LIMITS = {
    "A级": {"alert": 0.5, "action": 1},
    "B级": {"alert": 5, "action": 10},
    "C级": {"alert": 50, "action": 100},
    "D级": {"alert": 100, "action": 200},
}

# ============================
# 3. 工具函数
# ============================
def generate_sample_em_data():
    """生成模拟环境监测数据（30天，按级别随机）"""
    start = datetime(2026, 1, 1)
    dates = [start + timedelta(days=i) for i in range(30)]
    data = []
    for d in dates:
        grade = random.choice(["A级", "B级", "C级", "D级"])
        if grade == "A级":
            val = round(max(0, random.gauss(0.2, 0.15)), 2)
        elif grade == "B级":
            val = round(max(0, random.gauss(4, 3)), 2)
        elif grade == "C级":
            val = round(max(0, random.gauss(60, 30)), 2)
        else:  # D级
            val = round(max(0, random.gauss(150, 50)), 2)
        data.append({"日期": d, "洁净级别": grade, "浮游菌(cfu/m³)": val})
    return pd.DataFrame(data)

def generate_sample_endotoxin():
    """生成模拟内毒素标准曲线数据"""
    conc = [0.00, 0.05, 0.10, 0.25, 0.50, 1.00]
    od = [0.010, 0.085, 0.152, 0.340, 0.580, 0.890]
    return pd.DataFrame({"浓度(EU/mL)": conc, "OD值": od})

def linear_regression(x, y):
    """返回斜率、截距、R²"""
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    return slope, intercept, r2

def plot_em_trend(df, grade, limits):
    """绘制环境监测趋势图（带警戒/纠偏线）"""
    df_f = df[df["洁净级别"] == grade]
    if df_f.empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据", showarrow=False)
        return fig

    alert = limits["alert"]
    action = limits["action"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_f["日期"], y=df_f["浮游菌(cfu/m³)"],
        mode='lines+markers', name='监测值',
        line=dict(color='royalblue', width=2),
        marker=dict(size=8)
    ))
    fig.add_hline(y=alert, line_dash="dash", line_color="orange",
                  annotation_text=f"警戒限 {alert}", annotation_position="top right")
    fig.add_hline(y=action, line_dash="dash", line_color="red",
                  annotation_text=f"纠偏限 {action}", annotation_position="bottom right")
    fig.update_layout(
        title=f"{grade} 浮游菌趋势",
        xaxis_title="日期", yaxis_title="cfu/m³",
        height=450, template="plotly_white"
    )
    return fig

def plot_endotoxin_curve(df):
    """绘制内毒素标准曲线"""
    x = df["浓度(EU/mL)"].values
    y = df["OD值"].values
    slope, intercept, r2 = linear_regression(x, y)

    x_range = np.linspace(0, max(x)*1.1, 50)
    y_range = slope * x_range + intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers', name='实测值',
        marker=dict(size=12, color='red')
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=y_range, mode='lines', name=f'拟合线 (R²={r2:.4f})',
        line=dict(color='green', width=2)
    ))
    fig.update_layout(
        title=f"内毒素标准曲线 (Y = {slope:.3f}X + {intercept:.3f})",
        xaxis_title="浓度 (EU/mL)", yaxis_title="OD值 (405nm)",
        height=450, template="plotly_white"
    )
    return fig, slope, intercept, r2

def perform_sst(df, slope, intercept, r2):
    """系统适用性验证 (SST)"""
    checks = {}
    blank_row = df[df["浓度(EU/mL)"] == 0]
    if not blank_row.empty:
        blank_od = blank_row["OD值"].values[0]
        checks["空白OD值 < 0.050"] = {
            "value": f"{blank_od:.4f}",
            "pass": blank_od < 0.050
        }
    else:
        checks["空白OD值 < 0.050"] = {"value": "未检测", "pass": False}

    checks["决定系数 R² ≥ 0.980"] = {
        "value": f"{r2:.4f}",
        "pass": r2 >= 0.980
    }

    max_bias = 0.0
    if slope != 0 and len(df) > 2:
        y_pred = slope * df["浓度(EU/mL)"] + intercept
        for i, row in df.iterrows():
            if row["浓度(EU/mL)"] == 0:
                continue
            theo = row["OD值"]
            pred = slope * row["浓度(EU/mL)"] + intercept
            if theo != 0:
                bias = abs((theo - pred) / theo) * 100
                if bias > max_bias:
                    max_bias = bias
        checks["各点相对偏差 < 20% (最大偏差)"] = {
            "value": f"{max_bias:.2f}%",
            "pass": max_bias < 20.0
        }
    else:
        checks["各点相对偏差 < 20% (最大偏差)"] = {"value": "无法计算", "pass": False}

    overall = all([v["pass"] for k, v in checks.items() if "pass" in v])
    return checks, overall

# ============================
# 4. 数据初始化
# ============================
if "em_df" not in st.session_state:
    st.session_state.em_df = generate_sample_em_data()
if "endo_df" not in st.session_state:
    st.session_state.endo_df = generate_sample_endotoxin()

# ============================
# 5. 侧边栏（模块切换 + 限值参考）
# ============================
st.sidebar.title("📂 功能导航")
page = st.sidebar.radio(
    "选择模块",
    ["📈 环境监测趋势", "🧪 内毒素标准曲线"],
    index=0
)

# 显示当前模块
st.sidebar.markdown(f"**当前页面：** {page}")

st.sidebar.markdown("---")
st.sidebar.subheader("📋 警戒限 / 行动限参考")
st.sidebar.dataframe(
    pd.DataFrame(QC_LIMITS).T.rename_axis("级别").reset_index(),
    use_container_width=True,
    hide_index=True
)
st.sidebar.caption("注：行动限超过时需启动调查")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 重置模拟数据"):
    st.session_state.em_df = generate_sample_em_data()
    st.session_state.endo_df = generate_sample_endotoxin()
    st.sidebar.success("已重置！")
    st.rerun()

st.sidebar.caption("v1.0 | 数据只读 · 合规设计")

# ============================
# 6. 页面1：环境监测趋势
# ============================
if page == "📈 环境监测趋势":
    st.header("📈 洁净区环境监测趋势分析")

    uploaded_em = st.file_uploader("上传环境监测数据 (Excel/CSV)", type=["xlsx", "csv"], key="em_upload")
    if uploaded_em is not None:
        try:
            if uploaded_em.name.endswith('.csv'):
                df_new = pd.read_csv(uploaded_em)
            else:
                df_new = pd.read_excel(uploaded_em)
            if all(col in df_new.columns for col in ["日期", "洁净级别", "浮游菌(cfu/m³)"]):
                st.session_state.em_df = df_new
                st.success("文件加载成功！")
            else:
                st.error("列名不符，请确保包含：日期, 洁净级别, 浮游菌(cfu/m³)")
        except Exception as e:
            st.error(f"解析失败: {e}")

    df_em = st.session_state.em_df.copy()
    grades = sorted(df_em["洁净级别"].unique().tolist())
    selected_grade = st.selectbox("选择洁净级别", grades)

    if selected_grade:
        fig = plot_em_trend(df_em, selected_grade, QC_LIMITS[selected_grade])
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 查看原始监测数据（只读）"):
        st.dataframe(
            df_em[df_em["洁净级别"] == selected_grade],
            use_container_width=True,
            hide_index=True
        )

# ============================
# 7. 页面2：内毒素标准曲线
# ============================
else:  # 内毒素页面
    st.header("🧪 内毒素标准曲线拟合与系统验证 (SST)")

    uploaded_endo = st.file_uploader("上传标准曲线数据 (列名: 浓度(EU/mL), OD值)", type=["xlsx", "csv"], key="endo_upload")
    if uploaded_endo is not None:
        try:
            if uploaded_endo.name.endswith('.csv'):
                df_new = pd.read_csv(uploaded_endo)
            else:
                df_new = pd.read_excel(uploaded_endo)
            if all(col in df_new.columns for col in ["浓度(EU/mL)", "OD值"]):
                st.session_state.endo_df = df_new
                st.success("文件加载成功！")
            else:
                st.error("列名不符，请确保包含：浓度(EU/mL), OD值")
        except Exception as e:
            st.error(f"解析失败: {e}")

    df_endo_original = st.session_state.endo_df.copy()

    st.subheader("📄 原始数据（只读）")
    st.dataframe(df_endo_original, use_container_width=True, hide_index=True)

    # 异常值排除控制
    st.divider()
    st.subheader("🔍 数据点纳入/排除控制")
    st.caption("勾选表示将该点从计算中排除（原始数据保留）")

    exclude_flags = {}
    cols = st.columns(min(len(df_endo_original), 6))
    for idx, (i, row) in enumerate(df_endo_original.iterrows()):
        with cols[idx % len(cols)]:
            exclude_flags[i] = st.checkbox(
                f"排除 {row['浓度(EU/mL)']} EU/mL",
                key=f"exclude_{i}",
                value=False
            )

    excluded_indices = [i for i, flag in exclude_flags.items() if flag]
    df_filtered = df_endo_original[~df_endo_original.index.isin(excluded_indices)].copy()

    st.info(f"纳入计算的点数: **{len(df_filtered)}** / {len(df_endo_original)} (排除 {len(excluded_indices)} 个)")

    data_ok = len(df_filtered) >= 3
    if not data_ok:
        st.warning("⚠️ 有效数据点少于3个，无法进行回归，请取消部分排除。")

    if st.button("🚀 执行曲线拟合与系统验证", type="primary", disabled=not data_ok):
        x = df_filtered["浓度(EU/mL)"].values
        y = df_filtered["OD值"].values
        slope, intercept, r2 = linear_regression(x, y)

        col1, col2, col3 = st.columns(3)
        col1.metric("斜率 (Slope)", f"{slope:.4f}")
        col2.metric("截距 (Intercept)", f"{intercept:.4f}")
        col3.metric("决定系数 (R²)", f"{r2:.4f}")

        fig, _, _, _ = plot_endotoxin_curve(df_filtered)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("✅ 计算机化系统验证报告 (SST)")
        checks, overall = perform_sst(df_filtered, slope, intercept, r2)

        for check_name, result in checks.items():
            if result["pass"]:
                st.success(f"✔️ {check_name}: {result['value']} (合格)")
            else:
                st.error(f"❌ {check_name}: {result['value']} (不合格)")

        if overall:
            st.balloons()
            st.success("### 🎉 系统适用性验证：**通过 (PASS)**\n\n该批次曲线符合药典要求。")
        else:
            st.error("### 🛑 系统适用性验证：**失败 (FAIL)**\n\n请检查标准品制备或排除异常点后重试。")

        st.caption(f"📝 审计追踪：计算基于 {len(df_filtered)} 个数据点 (排除 {len(excluded_indices)} 个) | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("点击上方按钮生成曲线并进行系统适用性验证。")
