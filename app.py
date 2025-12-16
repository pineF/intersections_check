import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import ast
import osmnx as ox
import geopandas as gpd
import numpy as np
import os  # ファイル操作用に追加

# --- 定数設定 ---
RECOVERY_FILE = "recovery_data.csv"  # 自動保存用のファイル名

# ページ設定
st.set_page_config(layout="wide", page_title="位置情報修正ツール (Auto-Save)")

# --- データのロード関数 (リカバリー対応版) ---
# キャッシュは使わず、毎回最新の状態をチェックするように変更
def load_data(file_or_path):
    df = pd.read_csv(file_or_path)
    # パース処理
    if 'landmarks_with_intersections' in df.columns:
        df['landmarks_with_intersections'] = df['landmarks_with_intersections'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    if 'review_status' not in df.columns:
        df['review_status'] = 'Unchecked'
    return df

# --- 自動保存関数 ---
def auto_save(df):
    """変更があるたびに呼び出して、CSVに保存する"""
    df.to_csv(RECOVERY_FILE, index=False)
    # ローカル環境のコンソールで確認用
    # print(f"Auto-saved to {RECOVERY_FILE}")

# ...(OSMnx関数やSnap関数は変更なし)...
# get_osmnx_data や snap_to_node はそのまま使ってください
@st.cache_data(show_spinner=False)
def get_osmnx_data(lat, lon, dist, tolerance):
    # (中略: 元のコードと同じ)
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

def snap_to_node(clicked_lat, clicked_lon, nodes_gdf, threshold_deg=0.0001):
    # (中略: 元のコードと同じ)
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

def format_option(index, row):
    # (中略: 元のコードと同じ)
    status = row.get('review_status', 'Unchecked')
    name = row.get('name', '名称不明')
    icon = "⬜"
    if status == 'Modified': icon = "✏️"
    elif status == 'Confirmed': icon = "✅"
    return f"{icon} [{index}] {name}"


# --- メインロジック ---
def main():
    st.title("📍 位置情報修正ツール (自動保存機能付)")

    # ==========================================
    # 1. データの読み込みロジック (大幅変更)
    # ==========================================
    st.sidebar.header("📁 データ管理")

    # リカバリーファイルの存在チェック
    has_recovery = os.path.exists(RECOVERY_FILE)
    
    # 状態の初期化
    if 'df' not in st.session_state:
        # A. リカバリーファイルがある場合
        if has_recovery:
            st.toast("🔄 前回の作業データを復元しました！", icon="📂")
            st.session_state.df = load_data(RECOVERY_FILE)
            st.session_state.using_recovery = True
        # B. 新規の場合
        else:
            st.session_state.using_recovery = False

    # サイドバー表示
    if st.session_state.get('using_recovery'):
        st.sidebar.warning("⚠️ 自動保存されたデータを使用中")
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
            # 初回ロード時にも一応保存しておく
            auto_save(st.session_state.df)
            st.session_state.using_recovery = True
            st.rerun()

    df = st.session_state.df

    # ==========================================
    # 2. 保存ボタン (変更なしだがDL推奨)
    # ==========================================
    st.sidebar.markdown("---")
    st.sidebar.header("💾 手動保存")
    
    # 進捗表示
    total = len(df)
    done = len(df[df['review_status'] != 'Unchecked'])
    st.sidebar.progress(done / total if total > 0 else 0)
    st.sidebar.caption(f"進捗: {done} / {total}")

    csv_data = df.to_csv(index=False).encode('utf-8-sig')
    st.sidebar.download_button(
        label="最新CSVをダウンロード",
        data=csv_data,
        file_name="corrected_landmarks_v3.csv",
        mime="text/csv",
        type="primary",
        help="作業が完了したら必ずダウンロードしてください"
    )

    # 3. リスト選択・ナビゲーション (元のコードと同様)
    st.sidebar.markdown("---")
    
    # ...(以下、元のコードのフィルタリング処理などはそのまま)...
    show_unfinished_only = st.sidebar.checkbox("未完了のみ表示", value=False)
    if show_unfinished_only:
        filtered_indices = df[df['review_status'] == 'Unchecked'].index.tolist()
    else:
        filtered_indices = df.index.tolist()

    if not filtered_indices:
        st.sidebar.success("完了！")
        filtered_indices = df.index.tolist()

    options_dict = {format_option(i, df.iloc[i]): i for i in filtered_indices}
    
    current_idx = st.session_state.get('current_row_index', 0)
    if current_idx not in filtered_indices and filtered_indices:
        current_idx = filtered_indices[0]

    current_label = format_option(current_idx, df.iloc[current_idx])
    
    # ラベルが見つからない場合の安全策
    if current_label not in options_dict:
        # 辞書の最初のキーを使う
        current_label = list(options_dict.keys())[0]

    selected_label = st.sidebar.selectbox(
        "編集対象:",
        options=list(options_dict.keys()),
        index=list(options_dict.keys()).index(current_label)
    )
    
    row_index = options_dict[selected_label]

    if row_index != st.session_state.get('current_row_index'):
        st.session_state.current_row_index = row_index
        st.session_state.current_lm_index = 0
        st.session_state.temp_click = None
        st.session_state.current_osmnx_nodes = None
        st.rerun()

    # 次へ・前へボタン
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


    # データの準備
    row = df.iloc[row_index]
    landmarks = row['landmarks_with_intersections']
    
    if not landmarks:
        st.warning("データなし")
        return

    landmark_names = [lm.get('name', '不明') for lm in landmarks]
    if st.session_state.get('current_lm_index', 0) >= len(landmark_names):
        st.session_state.current_lm_index = 0

    # --- メイン画面 ---
    st.markdown("---")
    col_h, col_s = st.columns([3, 1])
    with col_h:
        st.markdown(f"## 🏠 {row.get('name')}")
    with col_s:
        # ステータス変更時の自動保存
        current_status = row.get('review_status', 'Unchecked')
        if current_status == 'Unchecked':
            if st.button("✅ 確認完了 (次へ)", type="primary", use_container_width=True):
                st.session_state.df.at[row_index, 'review_status'] = 'Confirmed'
                
                # ★ここで自動保存★
                auto_save(st.session_state.df)
                
                next_indices = [i for i in filtered_indices if i > row_index]
                if next_indices:
                    st.session_state.current_row_index = next_indices[0]
                st.session_state.temp_click = None
                st.rerun()
        elif current_status == 'Confirmed':
             if st.button("未確認に戻す"):
                st.session_state.df.at[row_index, 'review_status'] = 'Unchecked'
                auto_save(st.session_state.df) # ★保存
                st.rerun()
        else:
             st.info("修正済み")

    selected_lm_index = st.session_state.current_lm_index
    if len(landmark_names) > 1:
        selected_lm_index = st.radio("対象", range(len(landmark_names)), format_func=lambda x: landmark_names[x], horizontal=True)
        if selected_lm_index != st.session_state.current_lm_index:
            st.session_state.current_lm_index = selected_lm_index
            st.session_state.temp_click = None
            st.rerun()

    target_lm = landmarks[selected_lm_index]
    show_map_interface(row_index, selected_lm_index, target_lm, row)


# --- 地図と修正ロジック ---
try:
    @st.fragment
    def show_map_interface(row_index, selected_lm_index, target_lm, row):
        render_map_content(row_index, selected_lm_index, target_lm, row)
except AttributeError:
    def show_map_interface(row_index, selected_lm_index, target_lm, row):
        render_map_content(row_index, selected_lm_index, target_lm, row)

def render_map_content(row_index, selected_lm_index, target_lm, row):
    # データ再取得
    current_list = st.session_state.df.iloc[row_index]['landmarks_with_intersections']
    if selected_lm_index >= len(current_list): return

    target_lm = current_list[selected_lm_index]
    current_intersection = target_lm.get('nearest_intersection')

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("🛠️ 編集")
        edit_mode = st.radio("モード", ["交差点", "ランドマーク"], horizontal=True, label_visibility="collapsed")
        
        with st.expander("設定"):
            dist = st.slider("半径", 50, 300, 300)
            tol = st.number_input("誤差", 10)

        if edit_mode == "交差点":
            if current_intersection:
                st.code(f"{current_intersection['intersection_lat']:.5f}, {current_intersection['intersection_lon']:.5f}")
            if st.session_state.get('temp_click'):
                lat, lon = st.session_state.temp_click
                if st.button("更新する", type="primary"):
                    new_data = {
                        "intersection_lat": lat, "intersection_lon": lon,
                        "street_count": 99, "is_manual_fix": True
                    }
                    st.session_state.df.iloc[row_index]['landmarks_with_intersections'][selected_lm_index]['nearest_intersection'] = new_data
                    st.session_state.df.at[row_index, 'review_status'] = 'Modified'
                    
                    # ★ここで自動保存★
                    auto_save(st.session_state.df)
                    
                    st.session_state.temp_click = None
                    st.success("保存しました")
                    st.rerun()
        else:
            d_lat = st.session_state.temp_click[0] if st.session_state.get('temp_click') else target_lm['lat']
            d_lon = st.session_state.temp_click[1] if st.session_state.get('temp_click') else target_lm['lon']
            n_lat = st.number_input("Lat", value=d_lat, format="%.6f", key="nlat")
            n_lon = st.number_input("Lon", value=d_lon, format="%.6f", key="nlon")
            
            if st.button("更新する", type="primary"):
                st.session_state.df.iloc[row_index]['landmarks_with_intersections'][selected_lm_index]['lat'] = n_lat
                st.session_state.df.iloc[row_index]['landmarks_with_intersections'][selected_lm_index]['lon'] = n_lon
                st.session_state.df.at[row_index, 'review_status'] = 'Modified'
                
                # ★ここで自動保存★
                auto_save(st.session_state.df)
                
                st.session_state.temp_click = None
                st.success("保存しました")
                st.rerun()

        if st.session_state.get('temp_click'):
            if st.button("選択解除"):
                st.session_state.temp_click = None
                st.rerun()

    with col1:
        # (地図表示コードは変更なし)
        c_lat, c_lon = (st.session_state.temp_click if st.session_state.get('temp_click') 
                        else (current_intersection['intersection_lat'], current_intersection['intersection_lon']) if edit_mode == "交差点" and current_intersection 
                        else (target_lm['lat'], target_lm['lon']))
        m = folium.Map([c_lat, c_lon], zoom_start=18)
        
        with st.spinner('...'):
            nodes, edges, _ = get_osmnx_data(target_lm['lat'], target_lm['lon'], dist, tol)
            if nodes is not None: st.session_state.current_osmnx_nodes = nodes
            
        if edges is not None: folium.GeoJson(edges, style_function=lambda x: {'color':'#999', 'opacity':0.5}).add_to(m)
        if nodes is not None: 
            for _, n in nodes.iterrows(): 
                folium.CircleMarker([n.geometry.y, n.geometry.x], radius=7, color="#F0F", fill=True).add_to(m)

        folium.Marker([row['lat'], row['lng']], icon=folium.Icon(color="blue", icon="home")).add_to(m)
        folium.Marker([target_lm['lat'], target_lm['lon']], icon=folium.Icon(color="green", icon="flag")).add_to(m)
        if current_intersection: folium.Marker([current_intersection['intersection_lat'], current_intersection['intersection_lon']], icon=folium.Icon(color="red")).add_to(m)
        if st.session_state.get('temp_click'): folium.Marker(st.session_state.temp_click, icon=folium.Icon(color="orange")).add_to(m)

        map_data = st_folium(m, height=500, width="100%")
        if map_data and map_data['last_clicked']:
            rl, rln = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
            sl, sln, sn = snap_to_node(rl, rln, st.session_state.get('current_osmnx_nodes'))
            if st.session_state.get('temp_click') != (sl, sln):
                st.session_state.temp_click = (sl, sln)
                st.rerun()

    with col2:
        st.markdown("---")
        with st.expander("ゴミ箱"):
            if st.button("削除"):
                st.session_state.df.iloc[row_index]['landmarks_with_intersections'].pop(selected_lm_index)
                st.session_state.df.at[row_index, 'review_status'] = 'Modified'
                
                # ★ここでも自動保存★
                auto_save(st.session_state.df)
                
                st.rerun()

if __name__ == "__main__":
    main()
