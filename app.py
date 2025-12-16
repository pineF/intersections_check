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

    if uploaded_file is None:
        st.info("👈 左のサイドバーから、処理済みのCSVファイル (final_landmark_results.csv など) をアップロードしてください。")
        return

    # --- 2. データのロード ---
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        # 文字列をリストに戻す
        if 'landmarks_with_intersections' in df.columns:
            df['landmarks_with_intersections'] = df['landmarks_with_intersections'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        return df

    if 'df' not in st.session_state:
        st.session_state.df = load_data(uploaded_file)
    
    if st.sidebar.button("データをリセット/再読み込み"):
        st.session_state.df = load_data(uploaded_file)
        st.session_state.temp_click = None # リセット時に選択ピンも消す
        st.rerun()

    df = st.session_state.df

    # --- 3. 保存ボタン ---
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

    # --- 4. 選択処理 ---
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 編集対象の選択")

    # 行番号変更時に選択ピンをリセットするためのロジック
    if 'current_row_index' not in st.session_state:
        st.session_state.current_row_index = 0

    row_index = st.sidebar.number_input(
        "行番号 (Index)", 
        min_value=0, max_value=len(df)-1, value=st.session_state.current_row_index, step=1
    )

    # 行が変わったら選択中のピンをクリア
    if row_index != st.session_state.current_row_index:
        st.session_state.current_row_index = row_index
        st.session_state.temp_click = None
        st.rerun()

    row = df.iloc[row_index]
    landmarks = row['landmarks_with_intersections']

    if not isinstance(landmarks, list) or len(landmarks) == 0:
        st.warning(f"行 {row_index} にはランドマーク情報がありません。")
        return

    # ランドマーク選択
    landmark_names = [lm.get('name', '不明') for lm in landmarks]
    
    # ラジオボタンの状態管理（リセット用）
    if 'current_lm_index' not in st.session_state:
        st.session_state.current_lm_index = 0
        
    selected_lm_index = st.sidebar.radio(
        "修正するランドマーク", 
        range(len(landmark_names)), 
        format_func=lambda x: landmark_names[x]
    )

    # ランドマークが変わったら選択ピンをクリア
    if selected_lm_index != st.session_state.current_lm_index:
        st.session_state.current_lm_index = selected_lm_index
        st.session_state.temp_click = None
        st.rerun()

    target_lm = landmarks[selected_lm_index]
    current_intersection = target_lm.get('nearest_intersection')

    # --- 5. アクセス情報の表示 (New!) ---
    st.markdown("### 🚃 アクセス情報")
    if 'access' in row and pd.notna(row['access']):
        st.info(f"**{row['access']}**")
    else:
        st.caption("※ アクセス情報はありません")

    # --- 6. 地図と修正 ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"🗺️ {target_lm.get('name')}")
        
        # 地図の中心
        if st.session_state.get('temp_click'): # クリックした場所があればそこ中心
            center_lat = st.session_state.temp_click[0]
            center_lon = st.session_state.temp_click[1]
        elif current_intersection:
            center_lat = current_intersection['intersection_lat']
            center_lon = current_intersection['intersection_lon']
        else:
            center_lat = target_lm['lat']
            center_lon = target_lm['lon']

        m = folium.Map(location=[center_lat, center_lon], zoom_start=19)

        # A. 店舗（青）
        folium.Marker(
            [row['lat'], row['lng']], 
            popup="店舗", 
            icon=folium.Icon(color="blue", icon="home")
        ).add_to(m)

        # B. ランドマーク（緑）
        folium.Marker(
            [target_lm['lat'], target_lm['lon']], 
            tooltip=f"ランドマーク: {target_lm['name']}", 
            icon=folium.Icon(color="green", icon="flag")
        ).add_to(m)

        # C. 現在の登録交差点（赤）
        if current_intersection:
            folium.Marker(
                [current_intersection['intersection_lat'], current_intersection['intersection_lon']], 
                popup="現在の登録地",
                icon=folium.Icon(color="red", icon="exclamation-sign")
            ).add_to(m)
            
        # D. 【New!】クリックした修正候補地点（オレンジ）
        if st.session_state.get('temp_click'):
            folium.Marker(
                st.session_state.temp_click,
                popup="修正候補（ここにする？）",
                icon=folium.Icon(color="orange", icon="star")
            ).add_to(m)

        # 地図描画
        map_data = st_folium(m, height=500, width="100%")

        # クリックイベントの検知と保存
        # 地図がクリックされ、かつ「直前のクリック」と違う場所なら session_state に保存してリロード
        if map_data and map_data['last_clicked']:
            clicked_coords = (map_data['last_clicked']['lat'], map_data['last_clicked']['lng'])
            
            # まだ保存されていない、または場所が変わった場合のみ更新
            if st.session_state.get('temp_click') != clicked_coords:
                st.session_state.temp_click = clicked_coords
                st.rerun()

    with col2:
        st.subheader("🛠️ 修正パネル")
        
        # 現在の状態
        if current_intersection:
            if current_intersection.get('is_manual_fix'):
                st.success("🟢 手動修正済み")
            else:
                st.info("🤖 自動検出データ")
        else:
            st.error("❌ 交差点データなし")

        st.markdown("---")
        
        # 修正候補がある場合（地図をクリック済み）
        if st.session_state.get('temp_click'):
            lat, lon = st.session_state.temp_click
            
            st.markdown("##### 📍 修正候補（オレンジのピン）")
            st.code(f"Lat: {lat:.6f}\nLon: {lon:.6f}")
            
            # 更新ボタン
            if st.button("この位置で確定更新", type="primary"):
                # 更新データ作成
                new_intersection_data = {
                    "intersection_lat": lat,
                    "intersection_lon": lon,
                    "street_count": 99, 
                    "is_manual_fix": True
                }
                
                # データ更新
                st.session_state.df.iloc[row_index]['landmarks_with_intersections'][selected_lm_index]['nearest_intersection'] = new_intersection_data
                
                # 選択ピンをクリアしてリロード
                st.session_state.temp_click = None
                st.success("✅ 更新しました！")
                st.rerun()
                
            if st.button("キャンセル"):
                st.session_state.temp_click = None
                st.rerun()
                
        else:
            st.write("地図上で**「正しい交差点」**をクリックすると、ここにピンと更新ボタンが表示されます。")

if __name__ == "__main__":
    main()
