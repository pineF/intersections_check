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
PAGE_TITLE = "位置情報修正ツール (Final v9)"

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

# --- 座標パース関数 ---
def parse_lat_lon_string(coord_str):
    if not coord_str:
        return None, None
    try:
        coord_str = coord_str.replace('，', ',')
        parts = coord_str.split(',')
        if len(parts) != 2:
            return None, None
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        return lat, lon
    except ValueError:
        return None, None

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
    
    total = len(df)
    done = len(df[df['review_status'] == 'Confirmed'])
    if total > 0: st.sidebar.progress(done / total)
    st.sidebar.caption(f"完了数: {done} / {total}")

    csv_data = df.to_csv(index=False).encode('utf-8-sig')
    st.sidebar.download_button(
        "最新CSVをダウンロード", csv_data, "corrected_landmarks_v11.csv", "text/csv", type="primary"
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 編集対象")

    show_unfinished_only = st.sidebar.checkbox("未完了のみ表示", value=False)
    
    if show_unfinished_only:
        filtered_indices = df[df['review_status'] != 'Confirmed'].index.tolist()
    else:
        filtered_indices = df.index.tolist()

    if not filtered_indices:
        st.sidebar.success("🎉 全て完了しました！")
        filtered_indices = df.index.tolist()

    options_dict = {format_option(i, df.iloc[i]): i for i in filtered_indices}
    
    current_idx = st.session_state.get('current_row_index', 0)
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

    with col_s:
        current_status = row.get('review_status', 'Unchecked')
        
        if current_status == 'Confirmed':
            st.success("ステータス: ✅ 確認済")
            if st.button("未確認に戻す", use_container_width=True):
                st.session_state.df.at[row_index, 'review_status'] = 'Unchecked'
                auto_save(st.session_state.df)
                st.rerun()
        else:
            if current_status == 'Modified':
                st.info("ステータス: ✏️ 修正あり")
            else:
                st.info("ステータス: 未確認")
            
            if st.button("✅ 確認完了 (次へ)", type="primary", use_container_width=True):
                st.session_state.df.at[row_index, 'review_status'] = 'Confirmed'
                auto_save(st.session_state.df)
                
                next_indices = [i for i in filtered_indices if i > row_index]
                if next_indices:
                    st.session_state.current_row_index = next_indices[0]
                
                st.session_state.temp_click = None
                st.rerun()

    st.markdown("---")


    # ==========================================
    # 4. ランドマーク管理 (統合UI)
    # ==========================================
    
    # 選択肢の作成（既存のランドマーク名 + 新規追加）
    landmark_names = [f"{i+1}. {lm.get('name', '不明')}" for i, lm in enumerate(landmarks)]
    options = landmark_names + ["➕ 新規登録"]
    
    # セッションステートのインデックス管理
    if st.session_state.get('current_lm_index', 0) > len(landmarks): # 範囲外なら0に戻す
        st.session_state.current_lm_index = 0
        
    selected_option_index = st.radio(
        "編集または追加を選択:", 
        range(len(options)), 
        format_func=lambda x: options[x],
        horizontal=True,
        index=st.session_state.current_lm_index
    )
    
    if selected_option_index != st.session_state.current_lm_index:
        st.session_state.current_lm_index = selected_option_index
        st.session_state.temp_click = None # タブ切り替え時にクリック状態をリセット
        st.rerun()

    # --- 条件分岐: 新規登録モード vs 編集モード ---
    
    if selected_option_index == len(landmarks):
        # ==========================================
        # ケースA: 新規追加モード
        # ==========================================
        col_map, col_act = st.columns([2, 1])
        
        shop_lat = row.get('lat', 35.6812) if pd.notna(row.get('lat')) else 35.6812
        shop_lon = row.get('lng', 139.7671) if pd.notna(row.get('lng')) else 139.7671

        default_str = ""
        if st.session_state.get('temp_click'):
            default_str = f"{st.session_state.temp_click[0]:.6f}, {st.session_state.temp_click[1]:.6f}"

        with col_act:
            st.subheader("🆕 新規登録フォーム")
            st.markdown("地図をクリックするか、座標をカンマ区切りで入力してください。")
            
            # デフォルト名: 店舗名 + 連番
            next_num = len(landmarks) + 1
            new_name = st.text_input("ランドマーク名", value=f"{row.get('name', '店舗')} (LM{next_num})")
            
            coord_input = st.text_input(
                "座標 (Lat, Lon)", 
                value=default_str, 
                placeholder="例: 35.6895, 139.6917"
            )
            
            st.markdown("---")
            if st.button("追加登録する", type="primary", use_container_width=True):
                lat_val, lon_val = parse_lat_lon_string(coord_input)

                if lat_val is None or lon_val is None:
                    st.error("❌ 座標の形式が正しくありません。「35.123, 139.123」のようにカンマ区切りで入力するか、地図をクリックしてください。")
                else:
                    new_landmark = {
                        'name': new_name,
                        'lat': lat_val,
                        'lon': lon_val,
                        'nearest_intersection': None 
                    }
                    landmarks.append(new_landmark)
                    st.session_state.df.at[row_index, 'landmarks_with_intersections'] = landmarks
                    st.session_state.df.at[row_index, 'review_status'] = 'Modified'
                    auto_save(st.session_state.df)
                    st.session_state.temp_click = None
                    # 追加されたランドマーク（最後尾）を選択状態にする
                    st.session_state.current_lm_index = len(landmarks) - 1
                    st.success("追加しました！")
                    st.rerun()

        with col_map:
            m = folium.Map(location=[shop_lat, shop_lon], zoom_start=18)
            
            shop_name = row.get('name', '店舗')
            folium.Marker(
                [shop_lat, shop_lon], 
                tooltip=f"店舗: {shop_name}", 
                popup=shop_name,
                icon=folium.Icon(color="blue", icon="home")
            ).add_to(m)
            
            # 既存のランドマークも表示（位置関係把握のため）
            for lm in landmarks:
                folium.Marker(
                    [lm['lat'], lm['lon']],
                    tooltip=lm.get('name', '既存LM'),
                    icon=folium.Icon(color="gray", icon="flag")
                ).add_to(m)

            if st.session_state.get('temp_click'):
                folium.Marker(st.session_state.temp_click, popup="新規地点", icon=folium.Icon(color="orange", icon="star")).add_to(m)

            map_data = st_folium(m, height=500, width="100%")
            if map_data and map_data['last_clicked']:
                click_lat, click_lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
                if st.session_state.get('temp_click') != (click_lat, click_lon):
                    st.session_state.temp_click = (click_lat, click_lon)
                    st.rerun()

    else:
        # ==========================================
        # ケースB: 編集モード
        # ==========================================
        target_lm = landmarks[selected_option_index]
        show_map_interface(row_index, selected_option_index, target_lm, row)


# --- 地図インターフェース（編集用） ---
try:
    @st.fragment
    def show_map_interface(row_index, selected_lm_index, target_lm, row):
        render_map_content(row_index, selected_lm_index, target_lm, row)
except AttributeError:
    def show_map_interface(row_index, selected_lm_index, target_lm, row):
        render_map_content(row_index, selected_lm_index, target_lm, row)

def render_map_content(row_index, selected_lm_index, target_lm, row):
    current_list = st.session_state.df.iloc[row_index]['landmarks_with_intersections']
    
    # 削除操作などでインデックスがずれた場合のガード
    if selected_lm_index >= len(current_list):
        st.session_state.current_lm_index = 0
        st.rerun()
        return

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
            
            if st.session_state.get('temp_click'):
                d_lat, d_lon = st.session_state.temp_click
            else:
                d_lat, d_lon = target_lm['lat'], target_lm['lon']
            
            default_str = f"{d_lat:.6f}, {d_lon:.6f}"

            coord_input_lm = st.text_input(
                "座標 (Lat, Lon)", 
                value=default_str,
                key="lm_coord_input",
                placeholder="例: 35.6895, 139.6917"
            )
            
            if st.button("位置を更新", type="primary"):
                lat_val, lon_val = parse_lat_lon_string(coord_input_lm)
                
                if lat_val is None or lon_val is None:
                    st.error("❌ 座標の形式が正しくありません。")
                else:
                    st.session_state.df.iloc[row_index]['landmarks_with_intersections'][selected_lm_index]['lat'] = lat_val
                    st.session_state.df.iloc[row_index]['landmarks_with_intersections'][selected_lm_index]['lon'] = lon_val
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
        
        shop_name = row.get('name', '店舗')
        folium.Marker(
            [shop_lat, shop_lon], 
            tooltip=f"店舗: {shop_name}",
            popup=shop_name,
            icon=folium.Icon(color="blue", icon="home")
        ).add_to(m)
        
        # 編集対象
        lm_name = target_lm.get('name', 'ランドマーク')
        folium.Marker(
            [target_lm['lat'], target_lm['lon']], 
            tooltip=lm_name, 
            popup=lm_name,
            icon=folium.Icon(color="green", icon="flag")
        ).add_to(m)
        
        # 他のランドマーク（参考表示）
        for i, lm in enumerate(current_list):
            if i != selected_lm_index:
                folium.Marker(
                    [lm['lat'], lm['lon']],
                    tooltip=lm.get('name', '他LM'),
                    icon=folium.Icon(color="gray", icon="flag")
                ).add_to(m)

        if current_intersection:
            folium.Marker(
                [current_intersection['intersection_lat'], current_intersection['intersection_lon']], 
                popup="登録済み交差点",
                tooltip="登録済み交差点",
                icon=folium.Icon(color="red")
            ).add_to(m)
            
        if st.session_state.get('temp_click'):
            folium.Marker(st.session_state.temp_click, popup="修正候補", icon=folium.Icon(color="orange", icon="star")).add_to(m)

        map_data = st_folium(m, height=700, width="100%")
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
            if st.button("このランドマークを削除"):
                st.session_state.df.iloc[row_index]['landmarks_with_intersections'].pop(selected_lm_index)
                st.session_state.df.at[row_index, 'review_status'] = 'Modified'
                auto_save(st.session_state.df)
                st.session_state.current_lm_index = 0
                st.session_state.temp_click = None
                st.success("削除しました")
                st.rerun()

if __name__ == "__main__":
    main()
