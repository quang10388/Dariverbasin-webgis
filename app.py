from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from folium.raster_layers import ImageOverlay

import streamlit as st
import leafmap.foliumap as leafmap

# --- Cấu hình chung ---
DATA_DIR = Path(__file__).parent / "data"

# Tâm bản đồ khoảng lưu vực sông Đà (chỉnh nếu muốn)
DEFAULT_CENTER = [21.5, 104.5]  # [lat, lon]
DEFAULT_ZOOM = 7


# ---------------------------------------------------------------------
# Helper: vẽ raster (.tif) lên folium mà KHÔNG dùng localtileserver
# (dùng cho DEM, HWSD, LULC). Làm việc tốt cả trên Streamlit Cloud.
# ---------------------------------------------------------------------
def add_raster_overlay(
    m,
    raster_path: Path,
    layer_name: str,
    colormap: str = "viridis",
    opacity: float = 0.6,
    nodata: float | int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    max_size: int = 2000,
):
    """Đọc 1-band GeoTIFF và phủ lên bản đồ.

    - Tự downsample nếu raster quá lớn (max_size ~ số pixel theo chiều dài nhất).
    - nodata: giá trị cần làm trong suốt (ví dụ LULC ngoài lưu vực = 0).
    """
    raster_path = Path(raster_path)
    if not raster_path.exists():
        st.sidebar.warning(f"Không tìm thấy raster: {raster_path.name}")
        return

    try:
        import matplotlib.cm as cm
        import matplotlib.colors as colors
    except Exception as e:  # pragma: no cover
        st.sidebar.error(f"Thiếu matplotlib: {e}")
        return

    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width
        scale = max(height, width) / max_size if max(height, width) > max_size else 1.0

        if scale > 1.0:
            out_shape = (int(height / scale), int(width / scale))
            data = src.read(
                1,
                out_shape=out_shape,
                resampling=Resampling.bilinear,
            )
        else:
            data = src.read(1)

        bounds = src.bounds
        if nodata is None:
            nodata = src.nodata

    data = data.astype("float32")

    mask = np.zeros_like(data, dtype=bool)
    if nodata is not None:
        mask |= data == nodata
    mask |= ~np.isfinite(data)

    data = np.where(mask, np.nan, data)

    # Tự tính khoảng màu nếu chưa cho
    if vmin is None:
        vmin = float(np.nanpercentile(data, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(data, 98))

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(colormap)

    rgba = cmap(norm(data))  # (H, W, 4), float 0–1
    # Làm trong suốt vùng NaN
    rgba[..., 3] = np.where(np.isnan(data), 0.0, rgba[..., 3])

    img = (rgba * 255).astype("uint8")

    img_overlay = ImageOverlay(
        image=img,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=opacity,
        name=layer_name,
        interactive=True,
        cross_origin=False,
    )
    img_overlay.add_to(m)


# ---------------------------------------------------------------------
# Các lớp bản đồ
# ---------------------------------------------------------------------
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
            layer_name="Sông suối",
            style={"color": "blue", "weight": 1},
        )


def add_dem_soil_layers(m):
    """DEM & soil."""
    # DEM có thể là bản gốc hoặc bản đã giảm kích thước để web ( *_web.tif )
    dem_fp_web = DATA_DIR / "DEM_DaRiver_WGS84_web.tif"
    dem_fp_full = DATA_DIR / "DEM_DaRiver_WGS84.tif"
    dem_fp = dem_fp_web if dem_fp_web.exists() else dem_fp_full

    soil_fp = DATA_DIR / "Soil_HWSD_Dariver.tif"

    # DEM
    if st.sidebar.checkbox("DEM địa hình", value=False) and dem_fp.exists():
        add_raster_overlay(
            m,
            dem_fp,
            layer_name="DEM",
            colormap="terrain",
            opacity=0.6,
        )

    # HWSD
    if st.sidebar.checkbox("Bản đồ đất (HWSD)", value=False) and soil_fp.exists():
        add_raster_overlay(
            m,
            soil_fp,
            layer_name="Soil HWSD",
            colormap="viridis",
            opacity=0.6,
        )


def add_lulc_layers(m):
    """Lớp sử dụng đất / che phủ (LULC) theo năm."""
    st.sidebar.subheader("LULC theo năm")
    year = st.sidebar.selectbox(
        "Chọn năm LULC",
        options=["Không hiển thị", "2020", "2021", "2022", "2023", "2024"],
        index=0,
    )

    if year == "Không hiển thị":
        return

    tif_name = f"Phan_loai_{year}.tif"
    lulc_fp = DATA_DIR / tif_name

    if lulc_fp.exists():
        add_raster_overlay(
            m,
            lulc_fp,
            layer_name=f"LULC {year}",
            colormap="tab20",
            opacity=1.0,
            nodata=0,  # ngoài lưu vực = 0 → trong suốt
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
            style={"color": "cyan", "radius": 4, "fillColor": "cyan"},
            info_mode="on_click",
        )

    if st.sidebar.checkbox("Hồ chứa (TQ)", value=False) and res_cn.exists():
        m.add_vector(
            str(res_cn),
            layer_name="Hồ chứa Trung Quốc",
            style={"color": "darkcyan", "radius": 4, "fillColor": "darkcyan"},
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
        - Bật/tắt các lớp: ranh lưu vực, sông suối, DEM, soil, LULC.
        - Xem bản đồ hồ chứa, nhà máy thủy điện, trạm thủy văn.
        - Chọn năm LULC (2020–2024) để so sánh biến động sử dụng đất.
        """
    )

    # Khởi tạo bản đồ
    m = leafmap.Map(
        center=DEFAULT_CENTER,
        zoom=DEFAULT_ZOOM,
        draw_control=False,
        measure_control=True,
        fullscreen_control=True,
    )

    # Thứ tự: LULC (nền), DEM/Soil, đường biên, hồ chứa
    add_lulc_layers(m)
    add_dem_soil_layers(m)
    add_basin_layers(m)
    add_reservoir_hydro_layers(m)
    add_basemap_control(m)

    m.to_streamlit(height=750)


if __name__ == "__main__":
    main()
