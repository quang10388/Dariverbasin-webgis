from pathlib import Path
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from folium.raster_layers import ImageOverlay

import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from PIL import Image

# ===== BẢNG MÀU LULC (giống GEE) =====
LULC_CLASSES = {
    0: ("Loại khác", "#000000"),   # đen
    1: ("Mặt nước", "#1f78b4"),    # xanh dương
    2: ("Nông nghiệp", "#ffd92f"), # vàng
    3: ("Rừng", "#4daf4a"),        # xanh lá
    4: ("Dân cư", "#e41a1c"),      # đỏ
    5: ("Đất trống", "#bdbdbd"),   # xám
}

# --- Cấu hình chung ---
DATA_DIR = Path(__file__).parent / "data"
RES_PLOT_DIR = DATA_DIR / "reservoir_plots"
LULC_FIG_DIR = DATA_DIR / "LULC"


# Tâm bản đồ khoảng lưu vực sông Đà
DEFAULT_CENTER = [21.5, 104.5]  # [lat, lon]
DEFAULT_ZOOM = 7


# ---------------------------------------------------------------------
# Helper: vẽ raster (.tif) lên folium (DEM, HWSD)
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
    """Đọc 1-band GeoTIFF và phủ lên bản đồ với colormap liên tục."""
    raster_path = Path(raster_path)
    if not raster_path.exists():
        st.sidebar.warning(f"Không tìm thấy raster: {raster_path.name}")
        return

    import matplotlib.cm as cm
    import matplotlib.colors as colors

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

    if np.all(np.isnan(data)):
        st.sidebar.warning(f"{layer_name}: tất cả đều NaN – không hiển thị.")
        return

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
def add_categorical_raster_overlay(
    m,
    raster_path: Path,
    layer_name: str,
    nodata: float | int | None = None,
    opacity: float = 0.8,
    max_size: int = 2000,
):
    """
    Vẽ raster phân loại (ví dụ bản đồ đất HWSD) với màu rời rạc cho từng giá trị.

    Trả về:
        dict {giá_trị_pixel: "#rrggbb"} để dùng vẽ chú giải bên sidebar.
    """
    raster_path = Path(raster_path)
    if not raster_path.exists():
        st.sidebar.warning(f"Không tìm thấy raster: {raster_path.name}")
        return {}

    import matplotlib.cm as cm

    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width
        scale = max(height, width) / max_size if max(height, width) > max_size else 1.0

        if scale > 1.0:
            out_shape = (int(height / scale), int(width / scale))
            data = src.read(
                1,
                out_shape=out_shape,
                resampling=Resampling.nearest,  # giữ nguyên mã lớp, không nội suy
            )
        else:
            data = src.read(1)
        bounds = src.bounds
        if nodata is None:
            nodata = src.nodata

    data = data.astype("int64")

    # Xác định vùng nodata
    mask = ~np.isfinite(data)
    if nodata is not None:
        mask |= data == nodata

    valid = ~mask
    if not np.any(valid):
        st.sidebar.warning(f"{layer_name}: tất cả pixel đều là nodata – không hiển thị.")
        return {}

    # Các giá trị lớp (4285, 4287, 4404, 7001, ...)
    unique_vals = np.unique(data[valid])
    n_classes = len(unique_vals)

    # Colormap rời rạc: mỗi lớp một màu
    cmap = cm.get_cmap("tab20", n_classes)

    h, w = data.shape
    img = np.zeros((h, w, 4), dtype=np.uint8)

    # Dict để trả ra làm chú giải
    classes: dict[int, str] = {}

    for idx, val in enumerate(unique_vals):
        rgba = cmap(idx)  # (r, g, b, a) 0–1
        r, g, b, _ = (np.array(rgba) * 255).astype(np.uint8)
        a = int(255 * opacity)

        mask_val = data == val
        img[mask_val, 0] = r
        img[mask_val, 1] = g
        img[mask_val, 2] = b
        img[mask_val, 3] = a

        # Màu dạng #rrggbb để vẽ chú giải
        classes[int(val)] = f"#{r:02x}{g:02x}{b:02x}"

    # nodata → trong suốt
    img[mask, 3] = 0

    img_overlay = ImageOverlay(
        image=img,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=1.0,
        name=layer_name,
        interactive=True,
        cross_origin=False,
    )
    img_overlay.add_to(m)

    return classes


def add_lulc_overlay(
    m,
    raster_path: Path,
    layer_name: str,
    nodata: int | None = None,  # không mặc định 0 nữa
    opacity: float = 0.9,
    max_size: int = 2000,
):
    """Vẽ LULC với bảng màu rời rạc LULC_CLASSES."""
    raster_path = Path(raster_path)
    if not raster_path.exists():
        st.sidebar.warning(f"Không tìm thấy raster: {raster_path.name}")
        return

    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width
        # Giảm kích thước nếu raster quá lớn
        scale = max(height, width) / max_size if max(height, width) > max_size else 1.0

        if scale > 1.0:
            out_shape = (int(height / scale), int(width / scale))
            data = src.read(
                1,
                out_shape=out_shape,
                resampling=Resampling.nearest,  # giữ nguyên mã lớp
            )
        else:
            data = src.read(1)

        bounds = src.bounds
        # Nếu chưa truyền nodata thì đọc từ metadata của raster
        if nodata is None:
            nodata = src.nodata

    data = data.astype("int32")

    # Mask nodata và giá trị không hợp lệ
    mask = ~np.isfinite(data)
    if nodata is not None:
        mask |= data == nodata

    codes = sorted(LULC_CLASSES.keys())
    max_code = max(codes)

    # Mọi giá trị ngoài [0, max_code] đều đưa về 0
    data = np.where((data >= 0) & (data <= max_code), data, 0)
    # Các pixel nodata cũng đưa về 0
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

    # GIẢI PHÁP CHÍNH: lớp 0 (background) làm TRONG SUỐT
    # -> không còn ô vuông đen bao ngoài
    lut[0, 3] = 0

    img = lut[data]  # (H, W, 4)

    img_overlay = ImageOverlay(
        image=img,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=1.0,
        name=layer_name,
        interactive=True,
        cross_origin=False,
    )
    img_overlay.add_to(m)


# ---------------------------------------------------------------------
# Vector layers (lưu vực, sông suối, hồ chứa, nhà máy...)
# ---------------------------------------------------------------------
def add_basin_layers(m):
    # Lưu vực sông Đà
    basin_da_fp = DATA_DIR / "Da_River_Basin.gpkg"
    streams_fp = DATA_DIR / "Da_Streams.gpkg"

    # Lưu vực sông Hồng (file mới bạn đã thêm)
    basin_hong_fp = DATA_DIR / "Hong_River_Basin.gpkg"

    st.sidebar.subheader("Lưu vực & sông suối")

    # ---- Lưu vực sông Đà ----
    if st.sidebar.checkbox("Ranh lưu vực Đà", value=True) and basin_da_fp.exists():
        gdf = gpd.read_file(basin_da_fp)
        folium.GeoJson(
            gdf,
            name="Lưu vực sông Đà",
            style_function=lambda feat: {"color": "red", "weight": 2, "fillOpacity": 0},
        ).add_to(m)

    # ---- Lưu vực sông Hồng ----
    if st.sidebar.checkbox("Ranh lưu vực Hồng", value=False) and basin_hong_fp.exists():
        gdf_hong = gpd.read_file(basin_hong_fp)
        folium.GeoJson(
            gdf_hong,
            name="Lưu vực sông Hồng",
            style_function=lambda feat: {"color": "purple", "weight": 2, "fillOpacity": 0},
        ).add_to(m)

    # ---- Mạng sông chính (sông Đà) ----
    if st.sidebar.checkbox("Mạng sông chính (Đà)", value=True) and streams_fp.exists():
        gdf = gpd.read_file(streams_fp)
        folium.GeoJson(
            gdf,
            name="Sông suối Đà",
            style_function=lambda feat: {"color": "blue", "weight": 1},
        ).add_to(m)


def add_dem_soil_layers(m):
    """DEM & soil."""
    # DEM có thể là bản gốc hoặc bản đã giảm kích thước để web ( *_web.tif )
    dem_fp_web = DATA_DIR / "DEM_DaRiver_WGS84_web.tif"
    dem_fp_full = DATA_DIR / "DEM_DaRiver_WGS84.tif"
    dem_fp = dem_fp_web if dem_fp_web.exists() else dem_fp_full

    soil_fp = DATA_DIR / "Soil_HWSD_Dariver.tif"

    # DEM: dữ liệu liên tục → dùng add_raster_overlay như trước
    if st.sidebar.checkbox("DEM địa hình", value=False) and dem_fp.exists():
        add_raster_overlay(
            m,
            dem_fp,
            layer_name="DEM",
            colormap="terrain",
            opacity=0.6,
        )

    # Soil HWSD: raster phân loại → dùng add_categorical_raster_overlay
    soil_classes = {}
    if st.sidebar.checkbox("Bản đồ đất (HWSD)", value=False) and soil_fp.exists():
        soil_classes = add_categorical_raster_overlay(
            m,
            soil_fp,
            layer_name="Soil HWSD",
            opacity=0.8,
        )

    # Vẽ chú giải mã đất (giống kiểu Paletted của QGIS)
    if soil_classes:
        with st.sidebar.expander("Chú giải Soil HWSD"):
            st.write("Giá trị mã đất (HWSD):")
            for val in sorted(soil_classes.keys()):
                color = soil_classes[val]
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;margin-bottom:4px">
                        <div style="width:14px;height:14px;background:{color};
                                    border:1px solid #555;margin-right:6px"></div>
                        <span>{val}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def add_lulc_layers(m):
    st.sidebar.subheader("LULC theo năm")

    # ===== LULC lưu vực sông Đà =====
    year_da = st.sidebar.selectbox(
        "Chọn năm LULC (sông Đà)",
        options=["Không hiển thị", 2020, 2021, 2022, 2023, 2024],
        index=4,
    )

    any_lulc = False  # cờ để quyết định có vẽ chú giải hay không

    if year_da != "Không hiển thị":
        tif_name = f"Phan_loai_{year_da}.tif"
        lulc_fp = DATA_DIR / tif_name

        if not lulc_fp.exists():
            st.sidebar.warning(f"Không tìm thấy file {tif_name} trong thư mục data/")
        else:
            add_lulc_overlay(
                m,
                lulc_fp,
                layer_name=f"LULC sông Đà {year_da}",
                opacity=0.9,
            )
            any_lulc = True

    # ----- Ngăn cách trên sidebar -----
    st.sidebar.markdown("---")

    # ===== LULC lưu vực sông Hồng =====
    year_hong = st.sidebar.selectbox(
        "Chọn năm LULC (sông Hồng)",
        options=["Không hiển thị", 2015, 2020, 2025],
        index=2,  # mặc định chọn 2025
    )

    if year_hong != "Không hiển thị":
        hong_name = f"LULC_SongHong_{year_hong}.tif"
        lulc_hong_fp = DATA_DIR / hong_name

        if not lulc_hong_fp.exists():
            st.sidebar.warning(f"Không tìm thấy file {hong_name} trong thư mục data/")
        else:
            add_lulc_overlay(
                m,
                lulc_hong_fp,
                layer_name=f"LULC sông Hồng {year_hong}",
                opacity=0.9,
            )
            any_lulc = True

    # ===== CHÚ GIẢI CHUNG CHO LULC (nếu có lớp nào đang bật) =====
    if any_lulc:
        with st.sidebar.expander("Chú giải lớp phủ LULC"):
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


def add_reservoir_layers(m):
    """Hồ chứa & nhà máy thủy điện + trạm thủy văn.
    Hồ chứa TQ sẽ được dùng để bắt sự kiện click.
    """
    st.sidebar.subheader("Hồ chứa & Thủy điện")

    res_vn = DATA_DIR / "Reservoirs_Dariverbasin_Vietnam.gpkg"
    res_cn = DATA_DIR / "Reservoirs_Dariverbasin_China.gpkg"
    hyd_vn = DATA_DIR / "Location_hydropower_Dariverbasin_Vietnam.gpkg"
    hyd_cn = DATA_DIR / "Location_hydropower_Dariverbasin_China.gpkg"
    hydro_station = DATA_DIR / "Hydro_Station_Vietnam.gpkg"

    if st.sidebar.checkbox("Hồ chứa (VN)", value=False) and res_vn.exists():
        gdf_vn = gpd.read_file(res_vn)
        folium.GeoJson(
            gdf_vn,
            name="Hồ chứa VN",
            style_function=lambda feat: {"color": "cyan", "weight": 1, "fillOpacity": 0.5},
            tooltip=folium.GeoJsonTooltip(fields=["Name"], aliases=["Hồ chứa:"]),
        ).add_to(m)

    # Hồ chứa TQ (có click)
    gdf_cn = None
    if st.sidebar.checkbox("Hồ chứa (TQ)", value=True) and res_cn.exists():
        gdf_cn = gpd.read_file(res_cn)
        folium.GeoJson(
            gdf_cn,
            name="Hồ chứa TQ",
            style_function=lambda feat: {"color": "magenta", "weight": 1, "fillOpacity": 0.5},
            highlight_function=lambda feat: {"weight": 3, "color": "yellow"},
            tooltip=folium.GeoJsonTooltip(fields=["Name"], aliases=["Hồ chứa:"]),
        ).add_to(m)

    if st.sidebar.checkbox("Nhà máy thủy điện (VN)", value=False) and hyd_vn.exists():
        gdf = gpd.read_file(hyd_vn)
        folium.GeoJson(
            gdf,
            name="Nhà máy thủy điện VN",
        ).add_to(m)

    if st.sidebar.checkbox("Nhà máy thủy điện (TQ)", value=False) and hyd_cn.exists():
        gdf = gpd.read_file(hyd_cn)
        folium.GeoJson(
            gdf,
            name="Nhà máy thủy điện TQ",
        ).add_to(m)

    if st.sidebar.checkbox("Trạm thủy văn (VN)", value=False) and hydro_station.exists():
        gdf = gpd.read_file(hydro_station)
        folium.GeoJson(
            gdf,
            name="Trạm thủy văn VN",
        ).add_to(m)

    return gdf_cn  # dùng để lấy danh sách tên hồ


# ---------------------------------------------------------------------
# Phần hiển thị ảnh kết quả hồ chứa
# ---------------------------------------------------------------------
def get_available_reservoirs_from_plots():
    """Danh sách hồ có folder ảnh trong data/reservoir_plots."""
    if not RES_PLOT_DIR.exists():
        return []
    return sorted([p.name for p in RES_PLOT_DIR.iterdir() if p.is_dir()])


def show_reservoir_plots(res_name: str):
    """Hiển thị toàn bộ ảnh PNG trong folder data/reservoir_plots/<res_name>."""
    if not res_name:
        return
    folder = RES_PLOT_DIR / res_name
    if not folder.exists():
        st.warning(f"Không tìm thấy thư mục ảnh cho hồ **{res_name}** trong `data/reservoir_plots/`.")
        return

    st.markdown(f"### 📊 Kết quả phân tích cho hồ: **{res_name}**")

    img_files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    if not img_files:
        st.info("Thư mục không có file `.png` nào.")
        return

    # Hiển thị dạng 3 cột
    cols = st.columns(3)
    for i, fname in enumerate(sorted(img_files)):
        path = folder / fname
        with cols[i % 3]:
            st.image(Image.open(path), caption=fname, use_column_width=True)

def show_lulc_figures():
    """Hiển thị các hình ảnh tổng hợp LULC trong thư mục data/LULC."""
    folder = LULC_FIG_DIR
    if not folder.exists():
        st.info("Không tìm thấy thư mục `data/LULC`.")
        return

    img_files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    if not img_files:
        st.info("Thư mục `data/LULC` không có file `.png` nào.")
        return

    st.markdown("## 📈 Tổng hợp kết quả LULC toàn lưu vực")

    # Hiển thị dạng 2 cột cho dễ nhìn
    cols = st.columns(2)
    for i, fname in enumerate(sorted(img_files)):
        path = folder / fname
        with cols[i % 2]:
            st.image(str(path), caption=fname, use_column_width=True)


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
        * Khi **kích vào hồ chứa (TQ)** hoặc chọn trong danh sách → hiển thị bộ ảnh kết quả phân tích (AEV, time-series...).
        """
    )

    # ---------------- NỀN BẢN ĐỒ ----------------
    st.sidebar.subheader("Nền bản đồ")

    basemap_name = st.sidebar.selectbox(
        "Chọn nền bản đồ",
        options=["OpenStreetMap", "OpenTopoMap", "Esri.WorldImagery"],
        index=2,
    )

    if basemap_name == "OpenStreetMap":
        tiles = "OpenStreetMap"
        attr = None
    elif basemap_name == "OpenTopoMap":
        tiles = "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
        attr = "© OpenTopoMap contributors"
    else:  # Esri.WorldImagery
        tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        attr = "Tiles © Esri"

    m = folium.Map(
        location=DEFAULT_CENTER,
        zoom_start=DEFAULT_ZOOM,
        tiles=tiles,
        attr=attr,
        control_scale=True,
    )

    # Lớp nhãn Việt Nam (Hoàng Sa, Trường Sa...)
    if st.sidebar.checkbox("Bật lớp nhãn Việt Nam (Hoàng Sa, Trường Sa...)", value=False):
        vn_label_url = (
            "https://tiles.arcgis.com/tiles/EaQ3hSM51DBnlwMq/"
            "arcgis/rest/services/VietnamLabels/MapServer/tile/{z}/{y}/{x}"
        )
        folium.TileLayer(
            vn_label_url,
            name="Vietnam labels (Esri)",
            attr="Esri VietnamLabels",
            overlay=True,
            control=True,
        ).add_to(m)

    # Thứ tự vẽ lớp
    add_lulc_layers(m)
    add_dem_soil_layers(m)
    add_basin_layers(m)
    gdf_cn = add_reservoir_layers(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # ---------------- HIỂN THỊ MAP & BẮT SỰ KIỆN CLICK ----------------
    if "selected_reservoir" not in st.session_state:
        st.session_state.selected_reservoir = ""

    map_data = st_folium(
        m,
        width=None,
        height=700,
        returned_objects=["last_active_drawing"],
    )

    # Nếu click vào hồ chứa (TQ), lấy thuộc tính Name
    if map_data and map_data.get("last_active_drawing"):
        props = map_data["last_active_drawing"].get("properties", {})
        clicked_name = props.get("Name")  # chú ý: đúng tên trường trong GPKG
        if clicked_name:
            st.session_state.selected_reservoir = clicked_name

    # ---------------- SIDEBAR: CHỌN HỒ BẰNG LIST ----------------
    # (Phòng khi người dùng muốn chọn trực tiếp mà không cần click map)
    available_res = get_available_reservoirs_from_plots()
    if available_res:
        default_index = 0
        if st.session_state.selected_reservoir in available_res:
            default_index = available_res.index(st.session_state.selected_reservoir)
        selected_from_list = st.sidebar.selectbox(
            "Hoặc chọn hồ để xem ảnh:",
            options=available_res,
            index=default_index,
        )
        st.session_state.selected_reservoir = selected_from_list

    # ---------------- HIỂN THỊ ẢNH KẾT QUẢ ----------------
    st.markdown("---")
    if st.session_state.selected_reservoir:
        show_reservoir_plots(st.session_state.selected_reservoir)
    else:
        st.info(
            "👉 Hãy **click vào một hồ chứa (TQ)** trên bản đồ "
            "hoặc chọn từ danh sách bên trái để xem ảnh kết quả."
        )
        # ---------------- ẢNH TỔNG HỢP LULC ----------------
    with st.expander("📈 Xem các biểu đồ & bản đồ tổng hợp LULC", expanded=False):
        show_lulc_figures()



if __name__ == "__main__":
    main()
