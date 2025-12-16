import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import ast
import osmnx as ox
import geopandas as gpd
import numpy as np # 距離計算用に必要

# ページ設定
st.set_page_config(layout="wide", page_title="交差点修正ツール")

# --- データのロード関数 ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if 'landmarks_with_intersections' in df.columns:
        df['landmarks_with_intersections'] = df['landmarks_with_intersections'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    return df

# --- OSMnxデータ取得関数 ---
@st.cache_data(show_spinner=False)
def get_osmnx_data(lat, lon, dist, tolerance):
    try:
        G = ox.graph_from_point((lat, lon), dist=dist, network_type='drive')
        G_proj = ox.project_graph(G)
        G_cons = ox.consolidate_intersections(G_proj, tolerance=tolerance, rebuild_graph=True, dead_ends=False)
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(G_cons)
        gdf_nodes = gdf_nodes.to_crs(epsg=4326)
        gdf_edges = gdf_edges.to_crs(epsg=4326)
        return gdf_nodes, gdf_edges, None
    except Exception as e:
        return None, None, str(e)

# --- スナップ判定関数 (New!) ---
def snap_to_node(clicked_lat, clicked_lon, nodes_gdf, threshold_deg=0.0001):
    """
    クリック位置に近いノードがあれば、その座標を返す。
    threshold_deg: 吸着する距離の閾値（約10m程度）
    """
    if nodes_gdf is None or nodes_gdf.empty:
        return clicked_lat, clicked_lon, False

    # 全ノードとの距離を計算（簡易的なユークリッド距離）
    # ※厳密なメートル計算ではないですが、UI上の吸着判定には十分です
    distances = np.sqrt(
        (nodes_gdf.geometry.y - clicked_lat)**2 + 
        (nodes_gdf.geometry.x - clicked_lon)**2
    )
    
    min_dist_idx = distances.idxmin()
    min_dist = distances.min()

    # 閾値以内なら吸着
    if min_dist < threshold_deg:
        nearest_node = nodes_gdf.loc[min_dist_idx]
        return nearest_node.geometry.y, nearest_node.geometry.x, True
    
    return clicked_lat, clicked_lon, False


# --- メインロジック ---
def main():
    st.title("📍 位置情報 手動修正ツール (OSMnx連携 + Snap)")

    # 1. データ読み込み
    st.sidebar.header("📁 データ読み込み")
    uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

    if uploaded_file is None:
        st.info("👈 左のサイドバーから、CSVファイルをアップロードしてください。")
        return

    if 'df' not in st.session_state:
        st.session_state.df = load_data(uploaded_file)
    
    if st.sidebar.button("データをリセット/再読み込み"):
        st.session_state.df = load_data(uploaded_file)
        st.session_state.temp_click = None
        st.session_state.current_osmnx_nodes = None # キャッシュクリア
        st.rerun()

    df = st.session_state.df

    # 2. 保存ボタン
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

    # 3. 編集対象選択
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
        st.session_state.current_osmnx_nodes = None
        st.rerun()

    row = df.iloc[row_index]
    landmarks = row['landmarks_with_intersections']

    if not isinstance(landmarks, list) or len(landmarks) == 0:
        st.warning(f"行 {row_index} には有効なランドマーク情報がありません。")
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
        st.session_state.current_osmnx_nodes = None
        st.rerun()

    target_lm = landmarks[selected_lm_index]
    
    # 店舗情報
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

    show_map_interface(row_index, selected_lm_index, target_lm, row)


# --- 地図インターフェース ---
try:
    @st.fragment
    def show_map_interface(row_index, selected_lm_index, target_lm, row):
        render_map_content(row_index, selected_lm_index, target_lm, row)
except AttributeError:
    def show_map_interface(row_index, selected_lm_index, target_lm, row):
        render_map_content(row_index, selected_lm_index, target_lm, row)


def render_map_content(row_index, selected_lm_index, target_lm, row):
    # データ取得チェック
    current_list = st.session_state.df.iloc[row_index]['landmarks_with_intersections']
    if selected_lm_index >= len(current_list):
        st.error("データエラー: リセットしてください")
        return

    target_lm = current_list[selected_lm_index]
    current_intersection = target_lm.get('nearest_intersection')
    
    col1, col2 = st.columns([2, 1])
    
    # --- パネル ---
    with col2:
        st.subheader("🛠️ 修正パネル")
        edit_mode = st.radio("編集モード", ["交差点の位置", "ランドマーク自体の位置"], horizontal=True)
        st.markdown("---")

        with st.expander("🌐 交差点検索設定 (OSMnx)", expanded=True):
            osmnx_dist = st.slider("検索半径 (m)", 50, 500, 100, step=50)
            osmnx_tol = st.number_input("集約許容誤差", value=10, min_value=1, max_value=50)

        st.markdown("---")
        if edit_mode == "交差点の位置":
            st.markdown("**現在の登録交差点**")
            if current_intersection:
                status = "🟢 手動修正済" if current_intersection.get('is_manual_fix') else "🤖 自動検出"
                st.caption(f"ステータス: {status}")
                st.code(f"Lat: {current_intersection['intersection_lat']:.6f}\nLon: {current_intersection['intersection_lon']:.6f}")
        else:
            st.markdown("**現在のランドマーク位置**")
            st.code(f"Lat: {target_lm['lat']:.6f}\nLon: {target_lm['lon']:.6f}")

    # --- 地図 ---
    with col1:
        st.subheader(f"🗺️ 地図: {target_lm.get('name')}")
        
        # 中心決定
        if st.session_state.get('temp_click'):
            center_lat, center_lon = st.session_state.temp_click
        elif edit_mode == "交差点の位置" and current_intersection:
            center_lat = current_intersection['intersection_lat']
            center_lon = current_intersection['intersection_lon']
        else:
            center_lat, center_lon = target_lm['lat'], target_lm['lon']

        m = folium.Map(location=[center_lat, center_lon], zoom_start=18)

        # OSMnxデータの取得
        search_lat = target_lm['lat']
        search_lon = target_lm['lon']
        
        with st.spinner('交差点候補を検索中...'):
            nodes, edges, error = get_osmnx_data(search_lat, search_lon, osmnx_dist, osmnx_tol)
            
            # スナップ用にsession_stateに保存しておく
            if nodes is not None:
                st.session_state.current_osmnx_nodes = nodes
        
        if error:
            st.warning(f"OSMnxエラー: {error}")
        
        # 描画
        if edges is not None:
            folium.GeoJson(edges, style_function=lambda x: {'color': '#888888', 'weight': 2, 'opacity': 0.5}).add_to(m)

        if nodes is not None:
            for idx, node_row in nodes.iterrows():
                folium.CircleMarker(
                    location=[node_row.geometry.y, node_row.geometry.x],
                    radius=7,
                    color="#FF00FF", # マゼンタ
                    fill=True,
                    fill_color="#FF00FF",
                    fill_opacity=0.6,
                    tooltip="交差点候補 (クリックで吸着)"
                ).add_to(m)

        # マーカー
        folium.Marker([row['lat'], row['lng']], popup="店舗", icon=folium.Icon(color="blue", icon="home")).add_to(m)
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

        # クリックイベント
        map_data = st_folium(m, height=500, width="100%")

        if map_data and map_data['last_clicked']:
            raw_lat = map_data['last_clicked']['lat']
            raw_lon = map_data['last_clicked']['lng']
            
            # ★ここでスナップ処理を行う★
            snapped_lat, snapped_lon, is_snapped = snap_to_node(
                raw_lat, raw_lon, 
                st.session_state.get('current_osmnx_nodes')
            )
            
            new_coords = (snapped_lat, snapped_lon)
            
            # 前回と同じ座標でなければ更新
            if st.session_state.get('temp_click') != new_coords:
                st.session_state.temp_click = new_coords
                if is_snapped:
                    st.toast("🧲 交差点候補にスナップしました！") # 通知を出す
                st.rerun()

    # --- アクション ---
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
