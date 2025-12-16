import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import ast

# ページ設定
st.set_page_config(layout="wide", page_title="位置情報修正ツール (Full+Delete)")

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
    st.title("📍 位置情報 手動修正ツール")

    # 1. サイドバー（データ読み込み）
    st.sidebar.header("📁 データ読み込み")
    uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

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
        st.session_state.current_lm_index = 0 # 行が変わったらランドマーク選択もリセット
        st.session_state.temp_click = None
        st.rerun()

    row = df.iloc[row_index]
    landmarks = row['landmarks_with_intersections']

    # ランドマークリストが空、またはNoneの場合の処理
    if not isinstance(landmarks, list) or len(landmarks) == 0:
        st.warning(f"行 {row_index} には有効なランドマーク情報がありません（0件）。")
        # 店舗情報だけ表示して終了
        st.markdown("---")
        st.markdown(f"## 🏠 {row.get('name', '名称不明')}")
        return

    landmark_names = [lm.get('name', '不明') for lm in landmarks]
    
    if 'current_lm_index' not in st.session_state:
        st.session_state.current_lm_index = 0
    
    # 削除などでインデックスが範囲外になった場合の安全策
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

    # ターゲット特定
    target_lm = landmarks[selected_lm_index]
    
    # --- 店舗情報表示 ---
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


# --- 地図と修正パネルを表示する関数 ---
try:
    @st.fragment
    def show_map_interface(row_index, selected_lm_index, target_lm, row):
        render_map_content(row_index, selected_lm_index, target_lm, row)
except AttributeError:
    def show_map_interface(row_index, selected_lm_index, target_lm, row):
        render_map_content(row_index, selected_lm_index, target_lm, row)

def render_map_content(row_index, selected_lm_index, target_lm, row):
    # 最新の情報を取得（削除処理などで古くなっている可能性があるため再取得）
    current_list = st.session_state.df.iloc[row_index]['landmarks_with_intersections']
    
    # リストが空になっている場合などのガード
    if selected_lm_index >= len(current_list):
        st.error("データが削除されました。左サイドバーで行などを選択し直してください。")
        return

    target_lm = current_list[selected_lm_index]
    current_intersection = target_lm.get('nearest_intersection')
    
    col1, col2 = st.columns([2, 1])
    
    # --- 右カラム：修正パネル ---
    with col2:
        st.subheader("🛠️ 修正パネル")
        
        edit_mode = st.radio(
            "編集モード",
            ["交差点の位置", "ランドマーク自体の位置"],
            horizontal=True
        )

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

    # --- 左カラム：地図 ---
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

        m = folium.Map(location=[center_lat, center_lon], zoom_start=19)

        # マーカー
        folium.Marker([row['lat'], row['lng']], popup=f"店舗", icon=folium.Icon(color="blue", icon="home")).add_to(m)
        folium.Marker([target_lm['lat'], target_lm['lon']], tooltip=f"ランドマーク", icon=folium.Icon(color="green", icon="flag")).add_to(m)

        if current_intersection:
            folium.Marker(
                [current_intersection['intersection_lat'], current_intersection['intersection_lon']], 
                popup="現在の登録交差点", icon=folium.Icon(color="red", icon="exclamation-sign")
            ).add_to(m)
            
        if st.session_state.get('temp_click'):
            folium.Marker(
                st.session_state.temp_click, popup="修正候補", icon=folium.Icon(color="orange", icon="star")
            ).add_to(m)

        map_data = st_folium(m, height=500, width="100%")

        if map_data and map_data['last_clicked']:
            clicked_coords = (map_data['last_clicked']['lat'], map_data['last_clicked']['lng'])
            if st.session_state.get('temp_click') != clicked_coords:
                st.session_state.temp_click = clicked_coords
                st.rerun()

    # --- パネル下部：アクションボタン ---
    with col2:
        # 1. 更新アクション
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

        # 2. 削除アクション (新規追加)
        st.markdown("---")
        with st.expander("🗑️ データを削除する"):
            st.warning("このランドマーク自体が誤りである場合、リストから削除します。この操作は元に戻せません。")
            
            if st.button("このランドマークを削除", type="secondary"):
                # リストから該当インデックスの要素を削除
                current_list = st.session_state.df.iloc[row_index]['landmarks_with_intersections']
                current_list.pop(selected_lm_index)
                
                # データフレームに書き戻す（参照渡しで更新されているはずだが念のため）
                st.session_state.df.at[row_index, 'landmarks_with_intersections'] = current_list
                
                # 選択状態をリセット
                st.session_state.current_lm_index = 0
                st.session_state.temp_click = None
                
                st.success("削除しました。")
                st.rerun()

if __name__ == "__main__":
    main()
