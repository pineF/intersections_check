import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import ast

# ページ設定
st.set_page_config(layout="wide", page_title="交差点修正ツール (Upload版)")

def main():
    st.title("📍 交差点位置 手動修正ツール")

    # --- 1. ファイルアップロード ---
    st.sidebar.header("📁 データ読み込み")
    uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

    # ファイルがアップロードされていない場合
    if uploaded_file is None:
        st.info("👈 左のサイドバーから、処理済みのCSVファイル (final_landmark_results.csv など) をアップロードしてください。")
        return

    # --- 2. データのロードと初期化 ---
    # セッションステート（メモリ）にデータがない、または別のファイルがアップロードされた場合にロード
    # file_uploaderには `file_id` がないので、名前などで簡易判定するか、単純に毎回読み込む設計にします
    
    # データ読み込み関数
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        # 文字列として保存されているリストを復元
        if 'landmarks_with_intersections' in df.columns:
            df['landmarks_with_intersections'] = df['landmarks_with_intersections'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        return df

    # セッションステートの初期化（まだ読み込んでいない場合のみ）
    if 'df' not in st.session_state:
        st.session_state.df = load_data(uploaded_file)
    
    # リセットボタン（新しいファイルを読み直したい時など）
    if st.sidebar.button("データをリセット/再読み込み"):
        st.session_state.df = load_data(uploaded_file)
        st.rerun()

    df = st.session_state.df

    # --- 3. ダウンロードボタン (保存機能) ---
    st.sidebar.markdown("---")
    st.sidebar.header("💾 保存")
    
    # データフレームをCSV文字列に変換
    csv_data = df.to_csv(index=False).encode('utf-8-sig')
    
    st.sidebar.download_button(
        label="修正済みCSVをダウンロード",
        data=csv_data,
        file_name="corrected_landmarks.csv",
        mime="text/csv",
        type="primary"
    )

    # --- 4. 店舗・ランドマーク選択 ---
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 編集対象の選択")

    # 行番号選択
    row_index = st.sidebar.number_input(
        "行番号 (Index)", 
        min_value=0, 
        max_value=len(df)-1, 
        value=0, 
        step=1
    )
    
    row = df.iloc[row_index]
    landmarks = row['landmarks_with_intersections']

    # ランドマークがない場合
    if not isinstance(landmarks, list) or len(landmarks) == 0:
        st.warning(f"行 {row_index} にはランドマーク情報がありません。")
        return

    # ランドマーク選択
    landmark_names = [lm.get('name', '不明') for lm in landmarks]
    selected_lm_index = st.sidebar.radio(
        "修正するランドマーク", 
        range(len(landmark_names)), 
        format_func=lambda x: landmark_names[x]
    )
    
    target_lm = landmarks[selected_lm_index]
    current_intersection = target_lm.get('nearest_intersection')

    # --- 5. メイン画面：地図と修正 ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"🗺️ {target_lm.get('name')}")
        
        # 中心座標決定
        if current_intersection:
            center_lat = current_intersection['intersection_lat']
            center_lon = current_intersection['intersection_lon']
        else:
            center_lat = target_lm['lat']
            center_lon = target_lm['lon']

        m = folium.Map(location=[center_lat, center_lon], zoom_start=18)

        # マーカー配置
        folium.Marker(
            [row['lat'], row['lng']], 
            popup="店舗", 
            icon=folium.Icon(color="blue", icon="home")
        ).add_to(m)

        folium.Marker(
            [target_lm['lat'], target_lm['lon']], 
            tooltip=f"ランドマーク: {target_lm['name']}", 
            icon=folium.Icon(color="green", icon="flag")
        ).add_to(m)

        if current_intersection:
            folium.Marker(
                [current_intersection['intersection_lat'], current_intersection['intersection_lon']], 
                popup="現在の登録交差点",
                icon=folium.Icon(color="red", icon="exclamation-sign")
            ).add_to(m)

        # マップ描画とクリックイベント取得
        map_data = st_folium(m, height=500, width="100%")

    with col2:
        st.subheader("🛠️ 修正パネル")
        
        # 状態表示
        st.markdown("**現在のステータス:**")
        if current_intersection:
            if current_intersection.get('is_manual_fix'):
                st.success("🟢 手動修正済み")
            else:
                st.info("🤖 自動検出データ")
        else:
            st.error("❌ 交差点データなし")

        st.markdown("---")
