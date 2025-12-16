import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import ast

# ページ設定
st.set_page_config(layout="wide", page_title="交差点修正ツール (高速版)")

# --- データのロード関数 ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if 'landmarks_with_intersections' in df.columns:
        df['landmarks_with_intersections'] = df['landmarks_with_intersections'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    return df

# --- メインロジック ---
def main():
    st.title("📍 交差点位置 手動修正ツール")

    # 1. サイドバー（データ読み込み）
    st.sidebar.header("📁 データ読み込み")
    uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

    if uploaded_file is None:
        st.info("👈 左のサイドバーから、処理済みのCSVファイルをアップロードしてください。")
        return

    # データ初期化
    if 'df' not in st.session_state:
        st.session_state.df = load_data(uploaded_file)
    
    if st.sidebar.button("データをリセット/再読み込み"):
        st.session_state.df = load_data(uploaded_file)
        st.session_state.temp_click = None
        st.rerun()

    df = st.session_state.df

    # 2. サイドバー（保存ボタン）
    st.sidebar.markdown("---")
    st.sidebar.header("💾 保存")
    csv_data = df.to_csv(index=False).encode('utf-8-sig')
    st.sidebar.download_button(
        label="修正済みCSVをダウンロード",
        data=csv_data,
        file_name="corrected_landmarks.csv",
        mime="text/csv",
        type="primary"
    )

    # 3. サイドバー（選択処理）
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 編集対象の選択")

    if 'current_row_index' not in st.session_state:
        st.session_state.current_row_index = 0

    row_index = st.sidebar.number_input(
        "行番号 (Index)", 
        min_value=0, max_value=len(df)-1, value=st.session_state.current_row_index, step=1
    )

    # 行変更検知
    if row_index != st.session_state.current_row_index:
        st.session_state.current_row_index = row_index
        st.session_state.temp_click = None
        st.rerun()

    row = df.iloc[row_index]
    landmarks = row['landmarks_with_intersections']

    if not isinstance(landmarks, list) or len(landmarks) == 0:
        st.warning("ランドマーク情報がありません。")
        return

    landmark_names = [lm.get('name', '不明') for lm in landmarks]
    
    if 'current_lm_index' not in st.session_state:
        st.session_state.current_lm_index = 0
        
    selected_lm_index = st.sidebar.radio(
        "修正するランドマーク", 
        range(len(landmark_names)), 
        format_func=lambda x: landmark_names[x]
    )

    if selected_lm_index != st.session_state.current_lm_index:
        st.session_state.current_lm_index = selected_lm_index
        st.session_state.temp_click = None
        st.rerun()

    # ターゲット特定
    target_lm = landmarks[selected_lm_index]
    
    # --- 店舗情報表示 ---
    st.markdown("---")
    shop_name = row.get('name', '名称不明')
    st.markdown(f"## 🏠 {shop_name}")
    if 'access' in row and pd.notna(row['access']):
        st.info(f"🚃 **アクセス:** {row['access']}")
    else:
        st.caption("※ アクセス情報はありません")
    st.markdown("---")

    # ★重要★ 地図部分だけを切り出して、部分更新（fragment）にする
    show_map_interface(row_index, selected_lm_index, target_lm, row)

# --- 地図と修正パネルを表示する関数（ここだけリロードされる） ---
# try-exceptは、古いStreamlitを使っている場合のエラー回避用
try:
    @st.fragment  # Streamlit 1.37以上で使える神機能
    def show_map_interface(row_index, selected_lm_index, target_lm, row):
        render_map_content(row_index, selected_lm_index, target_lm, row)
except AttributeError:
    # 古いバージョンの場合は普通に関数を定義
    def show_map_interface(row_index, selected_lm_index, target_lm, row):
        render_map_content(row_index, selected_lm_index, target_lm, row)

def render_map_content(row_index, selected_lm_index, target_lm, row):
    # 最新の交差点情報を取得
    current_intersection = st.session_state.df.iloc[row_index]['landmarks_with_intersections'][selected_lm_index]['nearest_intersection']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"🗺️ 周辺地図: {target_lm.get('name')}")
        
        # 中心の決定
        if st.session_state.get('temp_click'):
            center_lat, center_lon = st.session_state.temp_click
        elif current_intersection:
            center_lat = current_intersection['intersection_lat']
            center_lon = current_intersection['intersection_lon']
        else:
            center_lat, center_lon = target_lm['lat'], target_lm['lon']

        m = folium.Map(location=[center_lat, center_lon], zoom_start=19)

        # マーカー類
        folium.Marker([row['lat'], row['lng']], popup=f"店舗", icon=folium.Icon(color="blue", icon="home")).add_to(m)
        folium.Marker([target_lm['lat'], target_lm['lon']], tooltip="ランドマーク", icon=folium.Icon(color="green", icon="flag")).add_to(m)

        if current_intersection:
            folium.Marker(
                [current_intersection['intersection_lat'], current_intersection['intersection_lon']], 
                popup="現在の登録地", icon=folium.Icon(color="red", icon="exclamation-sign")
            ).add_to(m)
            
        if st.session_state.get('temp_click'):
            folium.Marker(
                st.session_state.temp_click, popup="修正候補", icon=folium.Icon(color="orange", icon="star")
            ).add_to(m)

        # クリック取得
        map_data = st_folium(m, height=500, width="100%")

        if map_data and map_data['last_clicked']:
            clicked_coords = (map_data['last_clicked']['lat'], map_data['last_clicked']['lng'])
            if st.session_state.get('temp_click') != clicked_coords:
                st.session_state.temp_click = clicked_coords
                st.rerun() # ここでは「fragmentの中だけ」がリロードされる

    with col2:
        st.subheader("🛠️ 修正パネル")
        if current_intersection and current_intersection.get('is_manual_fix'):
            st.success("🟢 手動修正済み")
        else:
            st.info("🤖 自動検出データ")

        st.markdown("---")
        
        if st.session_state.get('temp_click'):
            lat, lon = st.session_state.temp_click
            st.markdown("##### 📍 修正候補")
            st.code(f"Lat: {lat:.6f}\nLon: {lon:.6f}")
            
            if st.button("この位置で確定更新", type="primary"):
                # データ更新
                new_data = {
                    "intersection_lat": lat, "intersection_lon": lon,
                    "street_count": 99, "is_manual_fix": True
                }
                st.session_state.df.iloc[row_index]['landmarks_with_intersections'][selected_lm_index]['nearest_intersection'] = new_data
                
                st.session_state.temp_click = None
                st.success("更新しました！")
                st.rerun() # ここもfragment内だけリロード
            
            if st.button("キャンセル"):
                st.session_state.temp_click = None
                st.rerun()
        else:
            st.write("地図をクリックしてピンを立ててください。")

if __name__ == "__main__":
    main()
