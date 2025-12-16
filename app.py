import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import ast
import osmnx as ox
import geopandas as gpd
import numpy as np
import os

# --- 定数設定 ---
RECOVERY_FILE = "recovery_data.csv"
PAGE_TITLE = "位置情報修正ツール (Final v3)"

# ページ設定
st.set_page_config(layout="wide", page_title=PAGE_TITLE)

# --- データのロード関数 ---
def load_data(file_or_path):
    df = pd.read_csv(file_or_path)
    
    if 'landmarks_with_intersections' in df.columns:
        df['landmarks_with_intersections'] = df['landmarks_with_intersections'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    
    if 'review_status' not in df.columns:
        df['review_status'] = 'Unchecked'
        
    return df

# --- 自動保存関数 ---
def auto_save(df):
    df.to_csv(RECOVERY_FILE, index=False)

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

# --- スナップ判定関数 ---
def snap_to_node(clicked_lat, clicked_lon, nodes_gdf, threshold_deg=0.0001):
    if nodes_gdf is None or nodes_gdf.empty:
        return clicked_lat, clicked_lon, False
    
    distances = np.sqrt(
        (nodes_gdf.geometry.y - clicked_lat)**2 + 
        (nodes_gdf.geometry.x - clicked_lon)**2
    )
    if distances.min() < threshold_deg:
        nearest_node = nodes_gdf.loc[distances.idxmin()]
        return nearest_node.geometry.y, nearest_node.geometry.x, True
    return clicked_lat, clicked_lon, False

# --- リスト表示用のフォーマット関数 ---
def format_option(index, row):
    status = row.get('review_status', 'Unchecked')
    name = row.get('name', '名称不明')
    icon = "⬜"
    if status == 'Modified': icon = "✏️"
    elif status == 'Confirmed': icon = "✅"
    return f"{icon} [{index}] {name}"


# --- メインロジック ---
def main():
    st.title("📍 ランドマーク＆交差点 修正ツール")

    # ==========================================
    # 1. データ読み込み
    # ==========================================
    st.sidebar.header("📁 データ管理")
    has_recovery = os.path.exists(RECOVERY_FILE)
    
    if 'df' not in st.session_state:
        if has_recovery:
            st.toast("🔄 作業データを復元しました", icon="📂")
            st.session_state.df = load_data(RECOVERY_FILE)
            st.session_state.using_recovery = True
        else:
            st.session_state.using_recovery = False

    if st.session_state.get('using_recovery'):
        st.sidebar.warning("⚠️ 自動保存データを使用中")
        if st.sidebar.button("🗑️ データを破棄してやり直す"):
            os.remove(RECOVERY_FILE)
            del st.session_state['df']
            st.session_state.using_recovery = False
            st.rerun()
    else:
        uploaded_file = st.sidebar.file_uploader("CSVアップロード", type=["csv"])
        if uploaded_file is None:
            st.info("👈 CSVファイルをアップロードしてください。")
            return
        
        if 'df' not in st.session_state:
            st.session_state.df = load_data(uploaded_file)
            auto_save(st.session_state.df)
            st.session_state.using_recovery = True
            st.rerun()

    df = st.session_state.df
    
    if 'review_status' not in df.columns:
        df['review_status'] = 'Unchecked'
        auto_save(df)
        st.rerun()

    # ==========================================
    # 2. サイドバー操作
    # ==========================================
    st.sidebar.markdown("---")
    
    # 進捗 (Confirmed以外は未完了扱い)
    total = len(df)
    done = len(df[df['review_status'] == 'Confirmed'])
    if total > 0: st.sidebar.progress(done / total)
    st.sidebar.caption(f"完了数: {done} / {total}")

    # 保存ボタン
    csv_data = df.to_csv(index=False).encode('utf-8-sig')
    st.sidebar.download_button(
        "最新CSVをダウンロード", csv_data, "corrected_landmarks_v5.csv", "text/csv", type="primary"
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 編集対象")

    # フィルタ (修正中も表示するように変更)
    show_unfinished_only = st.sidebar.checkbox("未完了のみ表示", value=False)
    
    if show_unfinished_only:
        # Confirmed 以外（Unchecked と Modified）を表示
        filtered_indices = df[df['review_status'] != 'Confirmed'].index.tolist()
    else:
        filtered_indices = df.index.tolist()

    if not filtered_indices:
        st.sidebar.success("🎉 全て完了しました！")
        filtered_indices = df.index.tolist()

    # リスト作成
    options_dict = {format_option(i, df.iloc[i]): i for i in filtered_indices}
    
    current_idx = st.session_state.get('current_row_index', 0)
    # フィルタ切り替え等で現在のインデックスがリストにない場合の救済
    if current_idx not in filtered_indices and filtered_indices:
        current_idx = filtered_indices[0]

    current_label = format_option(current_idx, df.iloc[current_idx])
    if current_label not in options_dict and options_dict:
        current_label = list(options_dict.keys())[0]

    if options_dict:
        selected_label = st.sidebar.selectbox(
            "リストから選択:",
            options=list(options_dict.keys()),
            index=list(options_dict.keys()).index(current_label)
        )
        row_index = options_dict[selected_label]
    else:
        row_index = 0

    if row_index != st.session_state.get('current_row_index'):
        st.session_state.current_row_index = row_index
        st.session_state.current_lm_index = 0
        st.session_state.temp_click = None
        st.session_state.current_osmnx_nodes = None
        st.rerun()

    c1, c2 = st.sidebar.columns(2)
    if c1.button("⬅️ 前へ"):
        prev_indices = [i for i in filtered_indices if i < row_index]
        if prev_indices:
            st.session_state.current_row_index = prev_indices[-1]
            st.session_state.temp_click = None
            st.rerun()
            
    if c2.button("次へ ➡️"):
        next_indices = [i for i in filtered_indices if i > row_index]
        if next_indices:
            st.session_state.current_row_index = next_indices[0]
            st.session_state.temp_click = None
            st.rerun()


    # ==========================================
    # 3. メインヘッダー
    # ==========================================
    row = df.iloc[row_index]
    landmarks = row['landmarks_with_intersections']
    if not isinstance(landmarks, list): landmarks = []

    st.markdown("---")
    col_h, col_s = st.columns([3, 1])
    
    with col_h:
        st.markdown(f"## 🏠 {row.get('name', '名称不明')}")
        if 'access' in row and pd.notna(row['access']):
            st.markdown(f"#### 🚃 {row['access']}")
        else:
            st.info("（案内文データなし）")

    # ★★★ ステータス管理ロジックの修正 ★★★
    with col_s:
        current_status = row.get('review_status', 'Unchecked')
        
        # 完了済みの場合
        if current_status == 'Confirmed':
            st.success("ステータス: ✅ 確認済")
            if st.button("未確認に戻す", use_container_width=True):
                st.session_state.df.at[row_index, 'review_status'] = 'Unchecked'
                auto_save(st.session_state.df)
                st.rerun()
        
        # 未確認 または 修正済みの場合
        else:
            if current_status == 'Modified':
                st.info("ステータス: ✏️ 修正あり")
            else:
                st.info("ステータス: 未確認")
            
            # どちらの状態でも「完了」ボタンを押せるようにする
            if st.button("✅ 確認完了 (次へ)", type="primary", use_container_width=True):
                st.session_state.df.at[row_index, 'review_status'] = 'Confirmed'
                auto_save(st.session_state.df)
                
                # 自動で次へ
                next_indices = [i for i in filtered_indices if i > row_index]
                if next_indices:
                    st.session_state.current_row_index = next_indices[0]
                
                st.session_state.temp_click = None
                st.rerun()

    st.markdown("---")


    # ==========================================
    # 4. コンテンツ分岐
    # ==========================================

    # --- ケースA: 新規作成 ---
    if len(landmarks) == 0:
        st.warning("⚠️ ランドマーク情報がありません。以下の地図をクリックして新規登録してください。")
        
        col_map, col_act = st.columns([2, 1])
        
        with col_act:
            st.subheader("🆕 新規登録")
            st.markdown("地図上で、店舗の入り口や目印となる場所をクリックしてください。")
            
            if st.session_state.get('temp_click'):
                lat, lon = st.session_state.temp_click
                st.code(f"Lat: {lat:.6f}\nLon: {lon:.6f}")
                
                new_name = st.text_input("ランドマーク名", value=row.get('name', '店舗前') + " (入口)")
                
                if st.button("この位置で登録する", type="primary"):
                    new_landmark = {
                        'name': new_name,
                        'lat': lat,
                        'lon': lon,
                        'nearest_intersection': None 
                    }
                    landmarks.append(new_landmark)
                    st.session_state.df.at[row_index, 'landmarks_with_intersections'] = landmarks
                    st.session_state.df.at[row_index, 'review_status'] = 'Modified'
                    auto_save(st.session_state.df)
                    st.session_state.temp_click = None
                    st.success("登録しました！続けて編集可能です。")
                    st.rerun()
            else:
                st.info("👈 地図をクリックしてください")

        with col_map:
            shop_lat = row.get('lat', 35.6812) if pd.notna(row.get('lat')) else 35.6812
            shop_lon = row.get('lng', 139.7671) if pd.notna(row.get('lng')) else 139.7671
            
            m = folium.Map(location=[shop_lat, shop_lon], zoom_start=18)
            folium.Marker([shop_lat, shop_lon], popup="店舗位置", icon=folium.Icon(color="blue", icon="home")).add_to(m)
            
            if st.session_state.get('temp_click'):
                folium.Marker(st.session_state.temp_click, popup="新規地点", icon=folium.Icon(color="orange", icon="star")).add_to(m)

            map_data = st_folium(m, height=500, width="100%")
            if map_data and map_data['last_clicked']:
                click_lat, click_lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
                if st.session_state.get('temp_click') != (click_lat, click_lon):
                    st.session_state.temp_click = (click_lat, click_lon)
                    st.rerun()

    # --- ケースB: 通常編集 ---
    else:
        landmark_names = [lm.get('name', '不明') for lm in landmarks]
        if st.session_state.get('current_lm_index', 0) >= len(landmark_names):
            st.session_state.current_lm_index = 0

        # タブ選択
        selected_lm_index = st.session_state.current_lm_index
        if len(landmark_names) > 1:
            selected_lm_index = st.radio(
                "編集するランドマークを選択", 
                range(len(landmark_names)), 
                format_func=lambda x: f"{x+1}. {landmark_names[x]}",
                horizontal=True,
                index=st.session_state.current_lm_index
            )
            if selected_lm_index != st.session_state.current_lm_index:
                st.session_state.current_lm_index = selected_lm_index
                st.session_state.temp_click = None
                st.rerun()

        target_lm = landmarks[selected_lm_index]
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
    current_list = st.session_state.df.iloc[row_index]['landmarks_with_intersections']
    if selected_lm_index >= len(current_list): return

    target_lm = current_list[selected_lm_index]
    current_intersection = target_lm.get('nearest_intersection')
    
    col1, col2 = st.columns([2, 1])
    
    # 操作パネル
    with col2:
        st.subheader("🛠️ 編集パネル")
        edit_mode = st.radio("編集対象", ["交差点の位置", "ランドマーク自体の位置"], horizontal=True)
        
        st.markdown("---")
        with st.expander("🌐 交差点検索設定 (OSMnx)", expanded=True):
            osmnx_dist = st.slider("検索半径 (m)", 50, 500, 300, step=50)
            osmnx_tol = st.number_input("許容誤差 (m)", min_value=1, value=10, step=1)

        st.markdown("---")

        if edit_mode == "交差点の位置":
            st.markdown("**現在の登録交差点**")
            if current_intersection:
                st.code(f"Lat: {current_intersection['intersection_lat']:.6f}\nLon: {current_intersection['intersection_lon']:.6f}")
            else:
                st.error("データなし")

            if st.session_state.get('temp_click'):
                lat, lon = st.session_state.temp_click
                st.markdown("##### 📍 更新候補")
                st.code(f"Lat: {lat:.6f}\nLon: {lon:.6f}")
                
                # 更新ボタン
                if st.button("この位置で更新", type="primary"):
                    new_data = {
                        "intersection_lat": lat, "intersection_lon": lon,
                        "street_count": 99, "is_manual_fix": True
                    }
                    st.session_state.df.iloc[row_index]['landmarks_with_intersections'][selected_lm_index]['nearest_intersection'] = new_data
                    st.session_state.df.at[row_index, 'review_status'] = 'Modified'
                    auto_save(st.session_state.df)
                    st.session_state.temp_click = None
                    st.success("更新しました！")
                    st.rerun()

        else:
            st.markdown("**ランドマーク位置**")
            d_lat = st.session_state.temp_click[0] if st.session_state.get('temp_click') else target_lm['lat']
            d_lon = st.session_state.temp_click[1] if st.session_state.get('temp_click') else target_lm['lon']
            
            new_lat = st.number_input("Lat", value=d_lat, format="%.6f", key="lm_lat_in")
            new_lon = st.number_input("Lon", value=d_lon, format="%.6f", key="lm_lon_in")
            
            # 更新ボタン
            if st.button("位置を更新", type="primary"):
                st.session_state.df.iloc[row_index]['landmarks_with_intersections'][selected_lm_index]['lat'] = new_lat
                st.session_state.df.iloc[row_index]['landmarks_with_intersections'][selected_lm_index]['lon'] = new_lon
                st.session_state.df.at[row_index, 'review_status'] = 'Modified'
                auto_save(st.session_state.df)
                st.session_state.temp_click = None
                st.success("更新しました！")
                st.rerun()

        if st.session_state.get('temp_click'):
            if st.button("選択解除", type="secondary"):
                st.session_state.temp_click = None
                st.rerun()

    # 地図
    with col1:
        if st.session_state.get('temp_click'):
            center_lat, center_lon = st.session_state.temp_click
        elif edit_mode == "交差点の位置" and current_intersection:
            center_lat, center_lon = current_intersection['intersection_lat'], current_intersection['intersection_lon']
        else:
            center_lat, center_lon = target_lm['lat'], target_lm['lon']

        m = folium.Map(location=[center_lat, center_lon], zoom_start=18)

        with st.spinner('交差点検索中...'):
            nodes, edges, err = get_osmnx_data(target_lm['lat'], target_lm['lon'], osmnx_dist, osmnx_tol)
            if nodes is not None: st.session_state.current_osmnx_nodes = nodes
        
        if edges is not None:
            folium.GeoJson(edges, style_function=lambda x: {'color': '#999', 'weight': 2, 'opacity': 0.5}).add_to(m)

        if nodes is not None:
            for _, n in nodes.iterrows():
                folium.CircleMarker([n.geometry.y, n.geometry.x], radius=6, color="#F0F", fill=True, tooltip="交差点").add_to(m)

        shop_lat = row.get('lat') if pd.notna(row.get('lat')) else center_lat
        shop_lon = row.get('lng') if pd.notna(row.get('lng')) else center_lon
        folium.Marker([shop_lat, shop_lon], popup="店舗", icon=folium.Icon(color="blue", icon="home")).add_to(m)
        folium.Marker([target_lm['lat'], target_lm['lon']], tooltip="LM", icon=folium.Icon(color="green", icon="flag")).add_to(m)

        if current_intersection:
            folium.Marker([current_intersection['intersection_lat'], current_intersection['intersection_lon']], popup="現在地", icon=folium.Icon(color="red")).add_to(m)
            
        if st.session_state.get('temp_click'):
            folium.Marker(st.session_state.temp_click, popup="修正候補", icon=folium.Icon(color="orange", icon="star")).add_to(m)

        map_data = st_folium(m, height=500, width="100%")
        if map_data and map_data['last_clicked']:
            rl, rln = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
            sl, sln, snapped = snap_to_node(rl, rln, st.session_state.get('current_osmnx_nodes'))
            if st.session_state.get('temp_click') != (sl, sln):
                st.session_state.temp_click = (sl, sln)
                if snapped: st.toast("🧲 Snap!")
                st.rerun()

    with col2:
        st.markdown("---")
        with st.expander("🗑️ 削除"):
            if st.button("削除実行"):
                st.session_state.df.iloc[row_index]['landmarks_with_intersections'].pop(selected_lm_index)
                st.session_state.df.at[row_index, 'review_status'] = 'Modified'
                auto_save(st.session_state.df)
                st.session_state.current_lm_index = 0
                st.session_state.temp_click = None
                st.rerun()

if __name__ == "__main__":
    main()
