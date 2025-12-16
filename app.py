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
RECOVERY_FILE = "recovery_data.csv"  # 自動保存用のファイル名
PAGE_TITLE = "位置情報修正ツール (Final Edition)"

# ページ設定
st.set_page_config(layout="wide", page_title=PAGE_TITLE)

# --- データのロード関数 ---
def load_data(file_or_path):
    df = pd.read_csv(file_or_path)
    
    # ランドマーク情報のパース (文字列 -> リスト/辞書)
    if 'landmarks_with_intersections' in df.columns:
        df['landmarks_with_intersections'] = df['landmarks_with_intersections'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    
    # ステータス管理用のカラムを追加 (なければ初期化)
    if 'review_status' not in df.columns:
        df['review_status'] = 'Unchecked'
        
    return df

# --- 自動保存関数 ---
def auto_save(df):
    """変更があるたびに呼び出して、CSVに保存する"""
    df.to_csv(RECOVERY_FILE, index=False)

# --- OSMnxデータ取得関数 (キャッシュ有効) ---
@st.cache_data(show_spinner=False)
def get_osmnx_data(lat, lon, dist, tolerance):
    try:
        # 指定された半径で道路ネットワークを取得
        G = ox.graph_from_point((lat, lon), dist=dist, network_type='drive')
        G_proj = ox.project_graph(G)
        # 交差点を集約
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
    
    icon = "⬜" # 未確認
    if status == 'Modified':
        icon = "✏️" # 修正済
    elif status == 'Confirmed':
        icon = "✅" # 確認済(OK)
        
    return f"{icon} [{index}] {name}"


# --- メインロジック ---
def main():
    st.title("📍 ランドマーク＆交差点 修正ツール (Final)")

    # ==========================================
    # 1. データ読み込み & リカバリー処理
    # ==========================================
    st.sidebar.header("📁 データ管理")

    # リカバリーファイルの存在チェック
    has_recovery = os.path.exists(RECOVERY_FILE)
    
    if 'df' not in st.session_state:
        # A. リカバリーファイルがある場合
        if has_recovery:
            st.toast("🔄 前回の作業データを復元しました", icon="📂")
            st.session_state.df = load_data(RECOVERY_FILE)
            st.session_state.using_recovery = True
        # B. 新規の場合
        else:
            st.session_state.using_recovery = False

    # サイドバー表示
    if st.session_state.get('using_recovery'):
        st.sidebar.warning("⚠️ 自動保存データを使用中")
        if st.sidebar.button("🗑️ 作業データを破棄してやり直す"):
            os.remove(RECOVERY_FILE)
            del st.session_state['df']
            st.session_state.using_recovery = False
            st.rerun()
    else:
        uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])
        if uploaded_file is None:
            st.info("👈 CSVファイルをアップロードしてください。")
            return
        
        if 'df' not in st.session_state:
            st.session_state.df = load_data(uploaded_file)
            auto_save(st.session_state.df) # 初回保存
            st.session_state.using_recovery = True
            st.rerun()

    df = st.session_state.df
    
    # 安全策: 列チェック
    if 'review_status' not in df.columns:
        df['review_status'] = 'Unchecked'
        # 保存してリロード
        auto_save(df)
        st.rerun()

    # ==========================================
    # 2. ダウンロードボタン
    # ==========================================
    st.sidebar.markdown("---")
    st.sidebar.header("💾 データ保存")
    
    # 進捗率
    total = len(df)
    done = len(df[df['review_status'] != 'Unchecked'])
    if total > 0:
        st.sidebar.progress(done / total)
    st.sidebar.caption(f"進捗: {done} / {total}")

    csv_data = df.to_csv(index=False).encode('utf-8-sig')
    st.sidebar.download_button(
        label="最新CSVをダウンロード",
        data=csv_data,
        file_name="corrected_landmarks_final.csv",
        mime="text/csv",
        type="primary"
    )

    # ==========================================
    # 3. リスト選択・ナビゲーション
    # ==========================================
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 編集対象")

    # フィルタリング
    show_unfinished_only = st.sidebar.checkbox("未完了のみ表示", value=False)
    
    if show_unfinished_only:
        filtered_indices = df[df['review_status'] == 'Unchecked'].index.tolist()
    else:
        filtered_indices = df.index.tolist()

    if not filtered_indices:
        st.sidebar.success("🎉 全て完了しました！")
        filtered_indices = df.index.tolist()

    # セレクトボックス用辞書
    options_dict = {format_option(i, df.iloc[i]): i for i in filtered_indices}
    
    # 現在のインデックス維持
    current_idx = st.session_state.get('current_row_index', 0)
    if current_idx not in filtered_indices and filtered_indices:
        current_idx = filtered_indices[0] # 見つからなければ先頭へ

    # 現在のラベル取得
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

    # 行変更検知
    if row_index != st.session_state.get('current_row_index'):
        st.session_state.current_row_index = row_index
        st.session_state.current_lm_index = 0
        st.session_state.temp_click = None
        st.session_state.current_osmnx_nodes = None
        st.rerun()

    # 前へ/次へボタン
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
    # 4. メインエリア表示ロジック
    # ==========================================
    row = df.iloc[row_index]
    landmarks = row['landmarks_with_intersections']
    
    # データ型チェック (NaNや文字列対策)
    if not isinstance(landmarks, list):
        landmarks = []

    # ---------------------------------------------------------
    # ケースA: ランドマーク情報がない場合 (新規作成モード)
    # ---------------------------------------------------------
    if len(landmarks) == 0:
        st.warning("⚠️ ランドマーク情報がありません。地図をクリックして新規登録してください。")
        
        col_map, col_act = st.columns([2, 1])
        
        with col_act:
            st.subheader("🆕 新規登録")
            st.markdown("地図上の、店舗の入り口や目印となる場所をクリックしてください。")
            
            if st.session_state.get('temp_click'):
                lat, lon = st.session_state.temp_click
                st.code(f"Lat: {lat:.6f}\nLon: {lon:.6f}")
                
                # 入力フォーム
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
                    
                    auto_save(st.session_state.df) # 保存
                    
                    st.session_state.temp_click = None
                    st.success("登録しました！")
                    st.rerun()
            else:
                st.info("👈 地図をクリックしてください")

        with col_map:
            # 店舗座標 (なければ東京駅)
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

    # ---------------------------------------------------------
    # ケースB: ランドマークがある場合 (通常編集モード)
    # ---------------------------------------------------------
    else:
        landmark_names = [lm.get('name', '不明') for lm in landmarks]
        
        if st.session_state.get('current_lm_index', 0) >= len(landmark_names):
            st.session_state.current_lm_index = 0

        # ヘッダー & 完了ボタン
        st.markdown("---")
        col_h, col_s = st.columns([3, 1])
        with col_h:
            st.markdown(f"## 🏠 {row.get('name', '名称不明')}")
            if 'access' in row and pd.notna(row['access']):
                st.caption(f"🚃 {row['access']}")
        
        with col_s:
            # ステータス表示と遷移ボタン
            current_status = row.get('review_status', 'Unchecked')
            if current_status == 'Unchecked':
                st.info("ステータス: 未確認")
                if st.button("✅ 確認完了 (次へ)", type="primary", use_container_width=True):
                    st.session_state.df.at[row_index, 'review_status'] = 'Confirmed'
                    auto_save(st.session_state.df) # 保存
                    
                    next_indices = [i for i in filtered_indices if i > row_index]
                    if next_indices:
                        st.session_state.current_row_index = next_indices[0]
                    st.session_state.temp_click = None
                    st.rerun()
            elif current_status == 'Confirmed':
                st.success("ステータス: ✅ 確認済")
                if st.button("未確認に戻す", use_container_width=True):
                    st.session_state.df.at[row_index, 'review_status'] = 'Unchecked'
                    auto_save(st.session_state.df) # 保存
                    st.rerun()
            else:
                st.success("ステータス: ✏️ 修正済")

        st.markdown("---")

        # 複数ランドマークがある場合のタブ選択
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


# --- 地図インターフェース (分離) ---
try:
    @st.fragment
    def show_map_interface(row_index, selected_lm_index, target_lm, row):
        render_map_content(row_index, selected_lm_index, target_lm, row)
except AttributeError:
    # 古いバージョンのStreamlit用
    def show_map_interface(row_index, selected_lm_index, target_lm, row):
        render_map_content(row_index, selected_lm_index, target_lm, row)

def render_map_content(row_index, selected_lm_index, target_lm, row):
    # 最新データを再取得
    current_list = st.session_state.df.iloc[row_index]['landmarks_with_intersections']
    if selected_lm_index >= len(current_list): return

    target_lm = current_list[selected_lm_index]
    current_intersection = target_lm.get('nearest_intersection')
    
    col1, col2 = st.columns([2, 1])
    
    # --- 右側パネル: 操作系 ---
    with col2:
        st.subheader("🛠️ 編集パネル")
        edit_mode = st.radio("編集対象", ["交差点の位置", "ランドマーク自体の位置"], horizontal=True)
        
        st.markdown("---")
        
        # OSMnx設定 (ご要望により 初期値300, Max500 に設定)
        with st.expander("🌐 交差点検索設定 (OSMnx)", expanded=True):
            osmnx_dist = st.slider("検索半径 (m)", min_value=50, max_value=500, value=300, step=50)
            osmnx_tol = st.number_input("集約許容誤差 (m)", value=10, min_value=1, max_value=50)

        st.markdown("---")

        # --- A. 交差点モード ---
        if edit_mode == "交差点の位置":
            st.markdown("**現在の登録交差点**")
            if current_intersection:
                st.caption("地図上のピンク色の丸をクリックして選択してください")
                st.code(f"Lat: {current_intersection['intersection_lat']:.6f}\nLon: {current_intersection['intersection_lon']:.6f}")
            else:
                st.error("交差点データなし")

            if st.session_state.get('temp_click'):
                lat, lon = st.session_state.temp_click
                st.markdown("##### 📍 更新候補")
                st.code(f"Lat: {lat:.6f}\nLon: {lon:.6f}")
                
                if st.button("交差点をこの位置で更新", type="primary"):
                    new_data = {
                        "intersection_lat": lat, "intersection_lon": lon,
                        "street_count": 99, 
                        "is_manual_fix": True
                    }
                    st.session_state.df.iloc[row_index]['landmarks_with_intersections'][selected_lm_index]['nearest_intersection'] = new_data
                    st.session_state.df.at[row_index, 'review_status'] = 'Modified'
                    
                    auto_save(st.session_state.df) # 保存
                    
                    st.session_state.temp_click = None
                    st.success("更新しました！")
                    st.rerun()

        # --- B. ランドマークモード ---
        else:
            st.markdown("**現在のランドマーク位置**")
            # 初期値設定（地図クリックがあればそちら優先）
            d_lat = st.session_state.temp_click[0] if st.session_state.get('temp_click') else target_lm['lat']
            d_lon = st.session_state.temp_click[1] if st.session_state.get('temp_click') else target_lm['lon']
            
            new_lat = st.number_input("緯度 (Lat)", value=d_lat, format="%.6f", key="lm_lat_in")
            new_lon = st.number_input("経度 (Lon)", value=d_lon, format="%.6f", key="lm_lon_in")
            
            if st.button("ランドマーク位置を更新", type="primary"):
                st.session_state.df.iloc[row_index]['landmarks_with_intersections'][selected_lm_index]['lat'] = new_lat
                st.session_state.df.iloc[row_index]['landmarks_with_intersections'][selected_lm_index]['lon'] = new_lon
                st.session_state.df.at[row_index, 'review_status'] = 'Modified'
                
                auto_save(st.session_state.df) # 保存
                
                st.session_state.temp_click = None
                st.success("更新しました！")
                st.rerun()

        # 共通キャンセルボタン
        if st.session_state.get('temp_click'):
            if st.button("選択解除", type="secondary"):
                st.session_state.temp_click = None
                st.rerun()

    # --- 左側パネル: 地図 ---
    with col1:
        # 中心の決定
        if st.session_state.get('temp_click'):
            center_lat, center_lon = st.session_state.temp_click
        elif edit_mode == "交差点の位置" and current_intersection:
            center_lat, center_lon = current_intersection['intersection_lat'], current_intersection['intersection_lon']
        else:
            center_lat, center_lon = target_lm['lat'], target_lm['lon']

        m = folium.Map(location=[center_lat, center_lon], zoom_start=18)

        # OSMnxデータ取得・描画
        with st.spinner('交差点候補を検索中...'):
            nodes, edges, err = get_osmnx_data(target_lm['lat'], target_lm['lon'], osmnx_dist, osmnx_tol)
            if nodes is not None:
                st.session_state.current_osmnx_nodes = nodes
        
        # エッジ(道路)
        if edges is not None:
            folium.GeoJson(edges, style_function=lambda x: {'color': '#999999', 'weight': 2, 'opacity': 0.5}).add_to(m)

        # ノード(交差点候補)
        if nodes is not None:
            for idx, node_row in nodes.iterrows():
                folium.CircleMarker(
                    location=[node_row.geometry.y, node_row.geometry.x],
                    radius=6, color="#FF00FF", fill=True, fill_color="#FF00FF", fill_opacity=0.5,
                    tooltip="交差点候補 (クリックで吸着)"
                ).add_to(m)

        # マーカー類
        # 店舗
        shop_lat = row.get('lat') if pd.notna(row.get('lat')) else center_lat
        shop_lon = row.get('lng') if pd.notna(row.get('lng')) else center_lon
        folium.Marker([shop_lat, shop_lon], popup="店舗", icon=folium.Icon(color="blue", icon="home")).add_to(m)
        
        # ランドマーク
        folium.Marker([target_lm['lat'], target_lm['lon']], tooltip="ランドマーク", icon=folium.Icon(color="green", icon="flag")).add_to(m)

        # 現在の交差点
        if current_intersection:
            folium.Marker(
                [current_intersection['intersection_lat'], current_intersection['intersection_lon']], 
                popup="登録済み交差点", icon=folium.Icon(color="red", icon="exclamation-sign")
            ).add_to(m)
            
        # クリック候補
        if st.session_state.get('temp_click'):
            folium.Marker(
                st.session_state.temp_click, popup="修正候補", icon=folium.Icon(color="orange", icon="star")
            ).add_to(m)

        # 地図表示 & クリック取得
        map_data = st_folium(m, height=500, width="100%")

        if map_data and map_data['last_clicked']:
            raw_lat = map_data['last_clicked']['lat']
            raw_lon = map_data['last_clicked']['lng']
            
            # 交差点吸着判定
            snapped_lat, snapped_lon, is_snapped = snap_to_node(
                raw_lat, raw_lon, st.session_state.get('current_osmnx_nodes')
            )
            
            new_coords = (snapped_lat, snapped_lon)
            if st.session_state.get('temp_click') != new_coords:
                st.session_state.temp_click = new_coords
                if is_snapped:
                    st.toast("🧲 交差点候補にスナップしました！")
                st.rerun()

    # --- 削除ボタン ---
    with col2:
        st.markdown("---")
        with st.expander("🗑️ このランドマークを削除"):
            if st.button("削除実行"):
                current_list = st.session_state.df.iloc[row_index]['landmarks_with_intersections']
                current_list.pop(selected_lm_index)
                st.session_state.df.at[row_index, 'landmarks_with_intersections'] = current_list
                st.session_state.df.at[row_index, 'review_status'] = 'Modified'
                
                auto_save(st.session_state.df) # 保存
                
                st.session_state.current_lm_index = 0
                st.session_state.temp_click = None
                st.success("削除しました。")
                st.rerun()

if __name__ == "__main__":
    main()
