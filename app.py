from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from folium.raster_layers import ImageOverlay

import streamlit as st
import leafmap.foliumap as leafmap


# ===== BẢNG MÀU LULC (giống GEE) =====
# Key = mã pixel trong file Phan_loai_20xx.tif
LULC_CLASSES = {
    1: ("Loại khác", "#000000"),   # đen
    2: ("Mặt nước", "#1f78b4"),    # xanh dương
    3: ("Nông nghiệp", "#ffd92f"), # vàng
    4: ("Rừng", "#4daf4a"),        # xanh lá
    5: ("Dân cư", "#e41a1c"),      # đỏ
    6: ("Đất trống", "#bdbdbd"),   # xám
}

# Danh sách màu theo thứ tự mã (nếu cần dùng nơi khác)
LULC_PALETTE = [LULC_CLASSES[k][1] for k in sorted(LULC_CLASSES.keys())]
LULC_VMIN = min(LULC_CLASSES.keys())
LULC_VMAX = max(LULC_CLASSES.keys())


# --- Cấu hình chung ---
DATA_DIR = Path(__file__).parent / "data"

# Tâm bản đồ khoảng lưu vực sông Đà (chỉnh nếu muốn)
DEFAULT_CENTER = [21.5, 104.5]  # [lat, lon]
DEFAULT_ZOOM = 7


# ---------------------------------------------------------------------
# Helper: vẽ raster (.tif) lên folium mà KHÔNG dùng localtileserver
# (dùng cho DEM, HWSD). Làm việc tốt cả trên Streamlit Cloud.
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
    - nodata: giá trị cần làm trong suốt.
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
# Helper riêng cho LULC: tô màu rời rạc theo LULC_CLASSES (không dùng localtileserver)
# ---------------------------------------------------------------------
def add_lulc_overlay(
    m,
    raster_path: Path,
    layer_name: str,
    nodata: int | None = 0,
    opacity: float = 0.9,
    max_size: int = 2000,
):
    """Vẽ LULC với bảng màu rời rạc LULC_CLASSES (không dùng localtileserver)."""
    raster_path = Path(raster_path)
    if not raster_path.exists():
        st.sidebar.warning(f"Không tìm thấy raster: {raster_path.name}")
        return

    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width
        scale = max(height, width) / max_size if max(height, width) > max_size else 1.0

        if scale > 1.0:
            out_shape = (int(height / scale), int(width / scale))
            data = src.read(
                1,
                out_shape=out_shape,
                resampling=Resampling.nearest,
            )
        else:
            data = src.read(1)
        bounds = src.bounds

    data = data.astype("int32")

    # Mask nodata và giá trị không hợp lệ
    mask = ~np.isfinite(data)
    if nodata is not None:
        mask |= data == nodata

    # Giá trị ngoài khoảng mã lớp → xem như nodata
    codes = sorted(LULC_CLASSES.keys())
    max_code = max(codes)
    data = np.where((data >= 0) & (data <= max_code), data, 0)
    data = np.where(mask, 0, data)

    # Bảng tra màu RGBA, index = mã lớp
    lut = np.zeros((max_code + 1, 4), dtype=np.uint8)
    for code in codes:
        _, hex_color = LULC_CLASSES[code]
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        lut[code, 0] = r
        lut[code, 1] = g
        lut[code, 2] = b
        lut[code, 3] = int(255 * opacity)

    img = lut[data]  # (H, W, 4)

    img_overlay = ImageOverlay(
        image=img,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=1.0,  # alpha đã nằm trong img
        name=layer_name,
        interactive=True,
        cross_origin=False,
    )
    img_overlay.add_to(m)


# ---------------------------------------------------------------------
# Các lớp bản đồ
# ---------------------------------------------------------------------
def add_basemap_control(m):
    """Chọn nền bản đồ & (tuỳ chọn) lớp nhãn Việt Nam."""
    st.sidebar.subheader("Nền bản đồ")

    # Nền chính: dùng các basemap có sẵn trong leafmap
    basemap_name = st.sidebar.selectbox(
        "Chọn nền bản đồ",
        options=["OpenStreetMap", "OpenTopoMap", "Esri.WorldImagery"],
        index=2,  # mặc định ảnh vệ tinh
    )
    m.add_basemap(basemap_name)

    # Lớp nhãn Việt Nam (có Hoàng Sa, Trường Sa, Biển Đông...)
    # Dữ liệu từ dịch vụ VietnamLabels của Esri
    if st.sidebar.checkbox("Bật lớp nhãn Việt Nam (Hoàng Sa, Trường Sa...)", value=False):
        vn_label_url = (
            "https://tiles.arcgis.com/tiles/EaQ3hSM51DBnlwMq/"
            "arcgis/rest/services/VietnamLabels/MapServer/tile/{z}/{y}/{x}"
        )
        m.add_tile_layer(
            vn_label_url,
            name="Vietnam labels (Esri)",
            attribution="Esri VietnamLabels",
            overlay=True,
            control=True,
        )


def add_basin_layers(m):
    """Ranh lưu vực & sông chính."""
    basin_fp = DATA_DIR / "Da_River_Basin.gpkg"
    streams_fp = DATA_DIR / "Da_Streams.gpkg"

    st.sidebar.subheader("Lưu vực & sông suối")

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
        options=["Không hiển thị", 2020, 2021, 2022, 2023, 2024],
        index=4,  # mặc định 2024
    )

    if year == "Không hiển thị":
        return

    tif_name = f"Phan_loai_{year}.tif"
    lulc_fp = DATA_DIR / tif_name

    if not lulc_fp.exists():
        st.sidebar.warning(f"Không tìm thấy file {tif_name} trong thư mục data/")
        return

    # Vẽ LULC với palette rời rạc giống GEE
    add_lulc_overlay(
        m,
        lulc_fp,
        layer_name=f"LULC {year}",
        nodata=0,           # 0 = ngoài lưu vực → trong suốt
        opacity=0.9,        # màu đậm, rõ
    )

    # Hiển thị chú giải nhỏ trong sidebar
    with st.sidebar.expander("Chú giải lớp phủ"):
        for code in sorted(LULC_CLASSES.keys()):
            name, color = LULC_CLASSES[code]
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;margin-bottom:4px">
                    <div style="width:14px;height:14px;background:{color};
                                border:1px solid #555;margin-right:6px"></div>
                    <span>{code}: {name}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def add_reservoir_hydro_layers(m):
    """Hồ chứa & nhà máy thủy điện + trạm thủy văn."""
    st.sidebar.subheader("Hồ chứa & Thủy điện")

    res_vn = DATA_DIR / "Reservoirs_Dariverbasin_Vietnam.gpkg"
    res_cn = DATA_DIR / "Reservoirs_Dariverbasin_China.gpkg"
    hyd_vn = DATA_DIR / "Location_hydropower_Dariverbasin_Vietnam.gpkg"
    hyd_cn = DATA_DIR / "Location_hydropower_Dariverbasin_China.gpkg"
    hydro_station = DATA_DIR / "Hydro_Station_Vietnam.gpkg"

    if st.sidebar.checkbox("Hồ chứa (VN)", value=False) and res_vn.exists():
        m.add_vector(
            str(res_vn),
            layer_name="Hồ chứa VN",
            style={"color": "cyan", "weight": 1, "fillOpacity": 0.5},
        )

    if st.sidebar.checkbox("Hồ chứa (TQ)", value=False) and res_cn.exists():
        m.add_vector(
            str(res_cn),
            layer_name="Hồ chứa TQ",
            style={"color": "magenta", "weight": 1, "fillOpacity": 0.5},
        )

    if st.sidebar.checkbox("Nhà máy thủy điện (VN)", value=False) and hyd_vn.exists():
        m.add_vector(
            str(hyd_vn),
            layer_name="Nhà máy thủy điện VN",
            style={"color": "orange", "radius": 4},
        )

    if st.sidebar.checkbox("Nhà máy thủy điện (TQ)", value=False) and hyd_cn.exists():
        m.add_vector(
            str(hyd_cn),
            layer_name="Nhà máy thủy điện TQ",
            style={"color": "purple", "radius": 4},
        )

    if st.sidebar.checkbox("Trạm thủy văn (VN)", value=False) and hydro_station.exists():
        m.add_vector(
            str(hydro_station),
            layer_name="Trạm thủy văn VN",
            style={"color": "blue", "radius": 4},
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="WebGIS trình diễn kết quả – Lưu vực sông Đà",
        layout="wide",
    )

    st.title("WebGIS trình diễn kết quả – Lưu vực sông Đà")

    st.markdown(
        """
        **Chức năng chính:**

        * Bật/tắt các lớp: ranh lưu vực, sông suối, DEM, soil, LULC.
        * Xem bản đồ hồ chứa, nhà máy thủy điện, trạm thủy văn.
        * Chọn năm LULC (2020–2024) để so sánh biến động sử dụng đất.
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
