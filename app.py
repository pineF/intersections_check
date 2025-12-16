import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import ast
import osmnx as ox
import geopandas as gpd

# ページ設定
st.set_page_config(layout="wide", page_title="交差点修正ツール (OSMnx版)")

# --- データのロード関数 ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if 'landmarks_with_intersections' in df.columns:
        df['landmarks_with_intersections'] = df['landmarks_with_intersections'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    return df

# --- OSMnxデータ取得関数 (キャッシュ化) ---
# 重たい処理なので、入力値が変わらない限り再計算しないようにキャッシュします
@st.cache_data(show_spinner=False)
def get_osmnx_data(lat, lon, dist, tolerance):
    try:
        # 1. グラフ取得
        G = ox.graph_from_point((lat, lon), dist=dist, network_type='drive')
        
        # 2. 投影変換 (メートル単位へ)
        G_proj = ox.project_graph(G)
        
        # 3. 交差点集約
        G_cons = ox.consolidate_intersections(G_proj, tolerance=tolerance, rebuild_graph=True, dead_ends=False)
        
        # 4. GeoDataFrame変換
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(G_cons)
        
        # 5. 地図表示用に緯度経度(EPSG:4326)に戻す
        gdf_nodes = gdf_nodes.to_crs(epsg=4326)
        gdf_edges = gdf_edges.to_crs(epsg=4326)
        
        return gdf_nodes, gdf_edges, None
    except Exception as e:
        return None, None, str(e)

# --- メインロジック ---
def main():
    st.title("📍 位置情報 手動修正ツール (OSMnx連携)")

    # 1. サイドバー（データ読み込み）
    st.sidebar.header("📁 データ読み込み")
    uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

    if uploaded_file is None:
        st.info("👈 左のサイドバーから、CSVファイルをアップロードしてください。")
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

    if row_index != st.session_state.current_row_index:
        st.session_state.current_row_index = row_index
        st.session_state.current_lm_index = 0
        st.session_state.temp_click = None
        st.rerun()

    row = df.iloc[row_index]
    landmarks = row['landmarks_with_intersections']

    if not isinstance(landmarks, list) or len(landmarks) == 0:
        st.warning(f"行 {row_index} には有効なランドマーク情報がありません。")
        st.markdown(f"## 🏠 {row.get('name', '名称不明')}")
        return

    landmark_names = [lm.get('name', '不明') for lm in landmarks]
    
    if 'current_lm_index' not in st.session_state:
        st.session_state.current_lm_index = 0
    if st.session_state.current_lm_index >= len(landmark_names):
        st.session_state.current_lm_index = 0

    selected_lm_index = st.sidebar.radio(
        "修正するランドマーク", 
        range(len(landmark_names)), 
        format_func=lambda x: landmark_names[x],
        index=st.session_state.current_lm_index
    )

    if selected_lm_index != st.session_state.current_lm_index:
        st.session_state.current_lm_index = selected_lm_index
        st.session_state.temp_click = None
        st.rerun()

    target_lm = landmarks[selected_lm_index]
    
    # 店舗情報表示
    st.markdown("---")
    shop_name = row.get('name', '名称不明')
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(f"## 🏠 {shop_name}")
        if 'access' in row and pd.notna(row['access']):
            st.info(f"🚃 **アクセス:** {row['access']}")
        else:
            st.caption("※ アクセス情報なし")
    
    st.markdown("---")

    # 地図インターフェース
    show_map_interface(row_index, selected_lm_index, target_lm, row)


# --- 地図描画ロジック ---
try:
    @st.fragment
    def show_map_interface(row_index, selected_lm_index, target_lm, row):
        render_map_content(row_index, selected_lm_index, target_lm, row)
except AttributeError:
    def show_map_interface(row_index, selected_lm_index, target_lm, row):
        render_map_content(row_index, selected_lm_index, target_lm, row)

def render_map_content(row_index, selected_lm_index, target_lm, row):
    # データ取得
    current_list = st.session_state.df.iloc[row_index]['landmarks_with_intersections']
    if selected_lm_index >= len(current_list):
        st.error("データエラー: リセットしてください")
        return

    target_lm = current_list[selected_lm_index]
    current_intersection = target_lm.get('nearest_intersection')
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("🛠️ 修正パネル")
        edit_mode = st.radio("編集モード", ["交差点の位置", "ランドマーク自体の位置"], horizontal=True)
        st.markdown("---")

        # --- OSMnx パラメータ設定 ---
        with st.expander("🌐 交差点検索設定 (OSMnx)", expanded=True):
            osmnx_dist = st.slider("検索半径 (m)", 50, 500, 100, step=50)
            osmnx_tol = st.number_input("集約許容誤差 (tolerance)", value=10, min_value=1, max_value=50)
            st.caption("※ 設定を変えると自動で再計算します")

        st.markdown("---")
        # 座標表示
        if edit_mode == "交差点の位置":
            st.markdown("**現在の登録交差点**")
            if current_intersection:
                status = "🟢 手動修正済" if current_intersection.get('is_manual_fix') else "🤖 自動検出"
                st.caption(f"ステータス: {status}")
                st.code(f"Lat: {current_intersection['intersection_lat']:.6f}\nLon: {current_intersection['intersection_lon']:.6f}")
            else:
                st.error("交差点データなし")
        else:
            st.markdown("**現在のランドマーク位置**")
            st.code(f"Lat: {target_lm['lat']:.6f}\nLon: {target_lm['lon']:.6f}")

    with col1:
        st.subheader(f"🗺️ 地図: {target_lm.get('name')}")
        
        # 中心の決定
        if st.session_state.get('temp_click'):
            center_lat, center_lon = st.session_state.temp_click
        elif edit_mode == "交差点の位置" and current_intersection:
            center_lat = current_intersection['intersection_lat']
            center_lon = current_intersection['intersection_lon']
        else:
            center_lat, center_lon = target_lm['lat'], target_lm['lon']

        m = folium.Map(location=[center_lat, center_lon], zoom_start=18)

        # --- A. OSMnxレイヤーの描画 ---
        # 検索中心点（基本はランドマークの位置、もしくは現在の交差点位置）
        search_lat = target_lm['lat']
        search_lon = target_lm['lon']
        
        with st.spinner('交差点候補を検索中...'):
            nodes, edges, error = get_osmnx_data(search_lat, search_lon, osmnx_dist, osmnx_tol)
        
        if error:
            st.warning(f"OSMnxエラー: {error}")
        
        if edges is not None:
            # 道路網（グレーの線）
            folium.GeoJson(
                edges,
                style_function=lambda x: {'color': '#888888', 'weight': 2, 'opacity': 0.5},
                name="道路網"
            ).add_to(m)

        if nodes is not None:
            # 交差点候補（マゼンタの円）
            # folium.GeoJsonだとクリックイベントが難しいので、CircleMarkerをループで追加する
            for idx, node_row in nodes.iterrows():
                folium.CircleMarker(
                    location=[node_row.geometry.y, node_row.geometry.x],
                    radius=6,
                    color="#FF00FF",      # マゼンタ（目立つ色）
                    fill=True,
                    fill_color="#FF00FF",
                    fill_opacity=0.6,
                    popup=f"交差点候補 (osmid: {idx})",
                    tooltip="交差点候補 (クリックで選択)"
                ).add_to(m)

        # --- B. 既存マーカーの描画 ---
        # 店舗（青）
        folium.Marker([row['lat'], row['lng']], popup="店舗", icon=folium.Icon(color="blue", icon="home")).add_to(m)
        # ランドマーク（緑）
        folium.Marker([target_lm['lat'], target_lm['lon']], tooltip="ランドマーク", icon=folium.Icon(color="green", icon="flag")).add_to(m)

        # 現在の交差点（赤）
        if current_intersection:
            folium.Marker(
                [current_intersection['intersection_lat'], current_intersection['intersection_lon']], 
                popup="現在の登録地", icon=folium.Icon(color="red", icon="exclamation-sign")
            ).add_to(m)
            
        # 修正候補（オレンジ）
        if st.session_state.get('temp_click'):
            folium.Marker(
                st.session_state.temp_click, popup="修正候補", icon=folium.Icon(color="orange", icon="star")
            ).add_to(m)

        # --- マップ描画とクリックイベント ---
        map_data = st_folium(m, height=500, width="100%")

        if map_data and map_data['last_clicked']:
            clicked_coords = (map_data['last_clicked']['lat'], map_data['last_clicked']['lng'])
            if st.session_state.get('temp_click') != clicked_coords:
                st.session_state.temp_click = clicked_coords
                st.rerun()

    # --- アクションボタン ---
    with col2:
        if st.session_state.get('temp_click'):
            lat, lon = st.session_state.temp_click
            
            st.markdown(f"##### 📍 修正候補 ({edit_mode})")
            st.code(f"Lat: {lat:.6f}\nLon: {lon:.6f}")
            
            if st.button("この位置で更新する", type="primary"):
                if edit_mode == "交差点の位置":
                    new_data = {
                        "intersection_lat": lat, "intersection_lon": lon,
                        "street_count": 99, "is_manual_fix": True
                    }
                    st.session_state.df.iloc[row_index]['landmarks_with_intersections'][selected_lm_index]['nearest_intersection'] = new_data
                    st.success("交差点位置を更新しました！")
                else:
                    st.session_state.df.iloc[row_index]['landmarks_with_intersections'][selected_lm_index]['lat'] = lat
                    st.session_state.df.iloc[row_index]['landmarks_with_intersections'][selected_lm_index]['lon'] = lon
                    st.success("ランドマーク位置を更新しました！")
                
                st.session_state.temp_click = None
                st.rerun()
            
            if st.button("キャンセル"):
                st.session_state.temp_click = None
                st.rerun()
        
        # 削除機能
        st.markdown("---")
        with st.expander("🗑️ データを削除する"):
            if st.button("このランドマークを削除", type="secondary"):
                current_list = st.session_state.df.iloc[row_index]['landmarks_with_intersections']
                current_list.pop(selected_lm_index)
                st.session_state.df.at[row_index, 'landmarks_with_intersections'] = current_list
                st.session_state.current_lm_index = 0
                st.session_state.temp_click = None
                st.success("削除しました。")
                st.rerun()

if __name__ == "__main__":
    main()
