from pathlib import Path

import streamlit as st
import leafmap.foliumap as leafmap

# --- Cấu hình chung ---
DATA_DIR = Path(__file__).parent / "data"

# Tâm bản đồ khoảng lưu vực sông Đà
DEFAULT_CENTER = [21.5, 104.5]  # [lat, lon]
DEFAULT_ZOOM = 7


def add_basemap_control(m: leafmap.Map) -> None:
    """Chọn nền bản đồ trong sidebar và thêm vào map."""
    st.sidebar.subheader("Nền bản đồ")

    basemap_options = {
        "Esri.WorldImagery (ảnh vệ tinh)": "Esri.WorldImagery",
        "OpenStreetMap": "OpenStreetMap",
        "CartoDB.Positron (sáng)": "CartoDB.Positron",
        "CartoDB.DarkMatter (tối)": "CartoDB.DarkMatter",
    }

    label = st.sidebar.selectbox(
        "Chọn nền bản đồ",
        options=list(basemap_options.keys()),
        index=0,
    )
    m.add_basemap(basemap_options[label])


def add_basin_layers(m: leafmap.Map) -> None:
    """Ranh lưu vực & mạng sông chính."""
    st.sidebar.subheader("Lưu vực & sông suối")

    basin_fp = DATA_DIR / "Da_River_Basin.gpkg"
    streams_fp = DATA_DIR / "Da_Streams.gpkg"

    if st.sidebar.checkbox("Ranh lưu vực Đà", value=True) and basin_fp.exists():
        m.add_vector(
            str(basin_fp),
            layer_name="Lưu vực sông Đà",
            style={"color": "red", "weight": 2, "fillOpacity": 0},
        )

    if st.sidebar.checkbox("Mạng sông chính", value=True) and streams_fp.exists():
        m.add_vector(
            str(streams_fp),
            layer_name="Sông chính",
            style={"color": "blue", "weight": 1},
        )


def add_dem_soil_layers(m: leafmap.Map) -> None:
    """DEM địa hình và bản đồ đất HWSD."""
    st.sidebar.subheader("DEM & bản đồ đất")

    dem_fp = DATA_DIR / "DEM_DaRiver_WGS84_web.tif"
    soil_fp = DATA_DIR / "Soil_HWSD_Dariver.tif"

    if st.sidebar.checkbox("DEM địa hình", value=True) and dem_fp.exists():
        m.add_raster(
            str(dem_fp),
            layer_name="DEM địa hình",
            colormap="terrain",
            opacity=0.7,
        )

    if st.sidebar.checkbox("Bản đồ đất (HWSD)", value=False) and soil_fp.exists():
        m.add_raster(
            str(soil_fp),
            layer_name="Bản đồ đất (HWSD)",
            colormap="viridis",
            opacity=0.8,
        )


def add_lulc_layers(m: leafmap.Map) -> None:
    """Lớp LULC theo các năm 2020–2024."""
    st.sidebar.subheader("LULC theo năm")

    year_options = ["Không hiển thị", 2020, 2021, 2022, 2023, 2024]
    year = st.sidebar.selectbox("Chọn năm LULC", options=year_options, index=0)

    if year == "Không hiển thị":
        return

    lulc_files = {
        2020: "Phan_loai_2020.tif",
        2021: "Phan_loai_2021.tif",
        2022: "Phan_loai_2022.tif",
        2023: "Phan_loai_2023.tif",
        2024: "Phan_loai_2024.tif",
    }

    tif_name = lulc_files.get(year)
    if tif_name is None:
        return

    lulc_fp = DATA_DIR / tif_name

    if lulc_fp.exists():
        # nodata=0: phần ngoài lưu vực (giá trị 0) sẽ trong suốt.
        m.add_raster(
            str(lulc_fp),
            layer_name=f"LULC {year}",
            colormap="tab20",
            opacity=1.0,
            nodata=0,
        )


def add_reservoir_hydro_layers(m: leafmap.Map) -> None:
    """Hồ chứa, nhà máy thủy điện và trạm thủy văn."""
    st.sidebar.subheader("Hồ chứa & Thủy điện")

    res_vn = DATA_DIR / "Reservoirs_Dariverbasin_Vietnam.gpkg"
    res_cn = DATA_DIR / "Reservoirs_Dariverbasin_China.gpkg"
    hyd_vn = DATA_DIR / "Location_hydropower_Dariverbasin_Vietnam.gpkg"
    hyd_cn = DATA_DIR / "Location_hydropower_Dariverbasin_China.gpkg"
    station_vn = DATA_DIR / "Hydro_Station_Vietnam.gpkg"

    if st.sidebar.checkbox("Hồ chứa (VN)", value=True) and res_vn.exists():
        m.add_vector(
            str(res_vn),
            layer_name="Hồ chứa VN",
            style={"color": "cyan", "radius": 5, "fillColor": "cyan"},
            info_mode="on_click",
        )

    if st.sidebar.checkbox("Hồ chứa (TQ)", value=False) and res_cn.exists():
        m.add_vector(
            str(res_cn),
            layer_name="Hồ chứa TQ",
            style={"color": "darkcyan", "radius": 5, "fillColor": "darkcyan"},
            info_mode="on_click",
        )

    if st.sidebar.checkbox("Nhà máy thủy điện (VN)", value=True) and hyd_vn.exists():
        m.add_vector(
            str(hyd_vn),
            layer_name="Thủy điện VN",
            style={"color": "yellow", "radius": 5, "fillColor": "yellow"},
            info_mode="on_click",
        )

    if st.sidebar.checkbox("Nhà máy thủy điện (TQ)", value=False) and hyd_cn.exists():
        m.add_vector(
            str(hyd_cn),
            layer_name="Thủy điện TQ",
            style={"color": "orange", "radius": 5, "fillColor": "orange"},
            info_mode="on_click",
        )

    if st.sidebar.checkbox("Trạm thủy văn (VN)", value=False) and station_vn.exists():
        m.add_vector(
            str(station_vn),
            layer_name="Trạm thủy văn VN",
            style={"color": "magenta", "radius": 5, "fillColor": "magenta"},
            info_mode="on_click",
        )


def main() -> None:
    """Hàm chính của ứng dụng."""
    st.set_page_config(
        page_title="WebGIS trình diễn kết quả – Lưu vực sông Đà",
        layout="wide",
    )

    st.title("WebGIS trình diễn kết quả – Lưu vực sông Đà")

    st.markdown(
        """
        **Chức năng chính:**

        * Bật/tắt các lớp: ranh lưu vực, sông suối, DEM, Soil, LULC.
        * Xem bản đồ hồ chứa, nhà máy thủy điện, trạm thủy văn.
        * Chọn năm LULC (2020–2024) để so sánh biến động sử dụng đất.
        """
    )

    # Tạo bản đồ chính
    m = leafmap.Map(
        center=DEFAULT_CENTER,
        zoom=DEFAULT_ZOOM,
        draw_control=False,
        measure_control=True,
        fullscreen_control=True,
    )

    # Thêm các lớp dữ liệu
    add_lulc_layers(m)
    add_basin_layers(m)
    add_dem_soil_layers(m)
    add_reservoir_hydro_layers(m)
    add_basemap_control(m)

    # Hiển thị lên Streamlit
    m.to_streamlit(height=750)


if __name__ == "__main__":
    main()
