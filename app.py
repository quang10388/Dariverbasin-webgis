from pathlib import Path

import streamlit as st
import leafmap.foliumap as leafmap

# --- Cấu hình chung ---
DATA_DIR = Path(__file__).parent / "data"

# Tâm bản đồ khoảng lưu vực sông Đà (chỉnh nếu muốn)
DEFAULT_CENTER = [21.5, 104.5]  # [lat, lon]
DEFAULT_ZOOM = 7


def add_basemap_control(m):
    """Chọn nền bản đồ trong sidebar và thêm vào map."""
    basemap_name = st.sidebar.selectbox(
        "Nền bản đồ",
        options=["OpenStreetMap", "OpenTopoMap", "Esri.WorldImagery"],
        index=2,  # mặc định ảnh vệ tinh
    )
    m.add_basemap(basemap_name)


def add_basin_layers(m):
    """Ranh lưu vực & sông chính."""
    dem_fp = DATA_DIR / "DEM_DaRiver_WGS84_web.tif"
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
            layer_name="Sông suối",
            style={"color": "blue", "weight": 1},
        )


def add_dem_soil_layers(m):
    """DEM & soil."""
    dem_fp = DATA_DIR / "DEM_DaRiver_WGS84_web.tif"
    soil_fp = DATA_DIR / "Soil_HWSD_Dariver.tif"

    if st.sidebar.checkbox("DEM địa hình", value=False) and dem_fp.exists():
        m.add_raster(
            str(dem_fp),
            layer_name="DEM",
            colormap="terrain",
            opacity=0.6,
        )

    if st.sidebar.checkbox("Bản đồ đất (HWSD)", value=False) and soil_fp.exists():
        m.add_raster(
            str(soil_fp),
            layer_name="Soil HWSD",
            colormap="viridis",
            opacity=0.6,
        )


def add_lulc_layers(m):
    st.sidebar.subheader("LULC theo năm")

    year = st.sidebar.selectbox(
        "Chọn năm LULC",
        options=["Không hiển thị", 2020, 2021, 2022, 2023, 2024],
        index=4,
    )
    if year == "Không hiển thị":
        return

    tif_name = f"Phan_loai_{year}.tif"
    lulc_fp = DATA_DIR / tif_name

    if lulc_fp.exists():
        m.add_raster(
            str(lulc_fp),
            layer_name=f"LULC {year}",
            colormap="tab20",   # bảng màu phân loại
            opacity=1.0,
            nodata=0,           # ngoài lưu vực = 0 → trong suốt
        )
    else:
        st.sidebar.warning(f"Không tìm thấy file {tif_name} trong thư mục data/")


def add_reservoir_hydro_layers(m):
    """Hồ chứa & nhà máy thủy điện + trạm thủy văn."""
    st.sidebar.subheader("Hồ chứa & Thủy điện")

    res_vn = DATA_DIR / "Reservoirs_Dariverbasin_Vietnam.gpkg"
    res_cn = DATA_DIR / "Reservoirs_Dariverbasin_China.gpkg"
    hyd_vn = DATA_DIR / "Location_hydropower_Dariverbasin_Vietnam.gpkg"
    hyd_cn = DATA_DIR / "Location_hydropower_Dariverbasin_China.gpkg"
    station_vn = DATA_DIR / "Hydro_Station_Vietnam.gpkg"

    if st.sidebar.checkbox("Hồ chứa (VN)", value=True) and res_vn.exists():
        m.add_vector(
            str(res_vn),
            layer_name="Hồ chứa Việt Nam",
            style={"color": "cyan", "radius": 6, "fillColor": "cyan"},
            info_mode="on_click",
        )

    if st.sidebar.checkbox("Hồ chứa (TQ)", value=False) and res_cn.exists():
        m.add_vector(
            str(res_cn),
            layer_name="Hồ chứa Trung Quốc",
            style={"color": "magenta", "radius": 6, "fillColor": "magenta"},
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

    if st.sidebar.checkbox("Trạm thủy văn (VN)", value=True) and station_vn.exists():
        m.add_vector(
            str(station_vn),
            layer_name="Trạm thủy văn VN",
            style={"color": "black", "radius": 5, "fillColor": "white"},
            info_mode="on_click",
        )


def main():
    st.set_page_config(
        page_title="WebGIS sông Đà – Hồ chứa & LULC",
        layout="wide",
    )

    st.title("WebGIS trình diễn kết quả – Lưu vực sông Đà")

    st.markdown(
        """
        **Chức năng chính:**
        - Bật/tắt các lớp: ranh lưu vực, sông suối, DEM, soil.
        - Xem bản đồ hồ chứa, nhà máy thủy điện, trạm thủy văn.
        - Chọn năm LULC (2020–2024) để so sánh biến động.
        """
    )

    m = leafmap.Map(
        center=DEFAULT_CENTER,
        zoom=DEFAULT_ZOOM,
        draw_control=False,
        measure_control=True,
        fullscreen_control=True,
    )

    # NỀN BẢN ĐỒ ĐƯỢC THÊM TRƯỚC
    add_basemap_control(m)

    # RASTER: LULC + DEM/SOIL
    add_lulc_layers(m)
    add_dem_soil_layers(m)

    # VECTOR: ranh lưu vực, hồ, thủy điện, trạm
    add_basin_layers(m)
    add_reservoir_hydro_layers(m)

    m.to_streamlit(height=750)


if __name__ == "__main__":
    main()
