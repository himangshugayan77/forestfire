import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from matplotlib import cm
from shapely.geometry import Point
import warnings

warnings.filterwarnings('ignore')

# Show full error details
st.set_option("client.showErrorDetails", True)

# Page layout
st.set_page_config(
    page_title="Assam Forest Fire Temperature Prediction Map",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

#===== CACHED DATA LOADERS =====
@st.cache_data(show_spinner=False)
def load_assam_districts():
    data = {
        'district': ['Guwahati', 'Dibrugarh', 'Jorhat', 'Silchar', 'Tezpur',
                    'Nagaon', 'Dhubri', 'Goalpara', 'Kokrajhar', 'Bongaigaon',
                    'Karimganj', 'Hailakandi', 'North Lakhimpur', 'Sivasagar',
                    'Golaghat', 'Morigaon', 'Darrang', 'Sonitpur'],
        'latitude': [26.1445, 27.4728, 26.7509, 24.8333, 26.6333,
                    26.3467, 26.0173, 26.1664, 26.4018, 26.4831,
                    24.8697, 24.6847, 27.2364, 26.9869, 26.7271,
                    26.2523, 26.4525, 26.6334],
        'longitude': [91.7362, 94.9120, 94.2037, 92.7789, 92.7833,
                      92.6811, 89.9583, 90.6167, 90.2631, 90.5436,
                      92.3542, 92.5442, 94.1181, 94.6851, 93.9615,
                      92.1738, 92.0219, 92.7833],
        'elevation': [55, 111, 116, 15, 58, 56, 37, 42, 46, 45, 8, 23, 105, 96, 295, 52, 61, 58],
        'forest_cover': [35.2, 62.8, 45.1, 28.4, 55.7, 38.9, 22.1, 41.3, 58.2, 33.7, 15.6, 32.1, 67.3, 52.4, 71.8, 29.5, 48.2, 55.7],
        'population': [957352, 154019, 153889, 228951, 58851, 141073, 71838, 58257, 60669, 75123, 79459, 59855, 26196, 114970, 122786, 55746, 69726, 58851]
    }
    df = pd.DataFrame(data)
    # Temperature sim
    base_temp = 30
    df['temperature'] = (
        base_temp + (df['latitude'] - df['latitude'].mean()) * -0.8
        + (df['elevation'] / 100) * -1.2
        + np.random.normal(0, 1.5, len(df))
    ).round(1)
    # Fire risk
    score = (
        (df['temperature'] - 25) * 0.3
        + (35 - df['forest_cover']) * 0.02
        + (df['elevation'] < 100).astype(int) * 2
        + np.random.normal(0, 0.5, len(df))
    )
    df['fire_risk'] = pd.cut(score, bins=[-np.inf,2,4,6,np.inf], labels=['Low','Medium','High','Very High'])
    # GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')
    return gdf

@st.cache_data(show_spinner=False)
def generate_temperature_grid(resolution=0.1):
    gdf = load_assam_districts()
    minx, miny, maxx, maxy = gdf.total_bounds
    lon_vals = np.arange(minx, maxx+resolution, resolution)
    lat_vals = np.arange(miny, maxy+resolution, resolution)
    pts, temps = [], []
    for lat in lat_vals:
        for lon in lon_vals:
            pts.append(Point(lon, lat))
            temps.append((25 + np.random.normal(0,5) + (lat-miny)*0.1).round(1))
    grid = gpd.GeoDataFrame({'temperature': temps, 'geometry': pts}, crs='EPSG:4326')
    return grid

#===== MAP CREATION =====
def create_interactive_map(districts, grid, show_grid, show_districts):
    center = [districts['latitude'].mean(), districts['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=7, tiles='OpenStreetMap')
    # Grid points
    if show_grid:
        temps = grid['temperature']
        tmin, tmax = temps.min(), temps.max()
        for idx, row in grid.sample(frac=0.05, random_state=1).iterrows():
            norm = (row.temperature - tmin)/(tmax-tmin)
            c = cm.coolwarm(norm)
            hexc = f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=3, color=hexc, fill=True, fillColor=hexc,
                fillOpacity=0.6, popup=f"{row.temperature}°C"
            ).add_to(m)
    # Districts
    if show_districts:
        colors={'Low':'green','Medium':'orange','High':'red','Very High':'darkred'}
        for _, r in districts.iterrows():
            col = colors.get(r.fire_risk, 'gray')
            popup = (
                f"<b>{r.district}</b><br>Temp: {r.temperature}°C<br>Risk: {r.fire_risk}"
            )
            folium.Marker(location=[r.latitude, r.longitude], popup=popup, icon=folium.Icon(color=col)).add_to(m)
    return m

#===== STREAMLIT LAYOUT =====
st.title("🔥 Assam Forest Fire Temperature Prediction Map")
st.markdown("Advanced interactive mapping for fire risk assessment in Assam")
# Sidebar controls
show_grid = st.sidebar.checkbox("Show Temperature Grid", True)
show_districts = st.sidebar.checkbox("Show District Markers", True)
resolution = st.sidebar.slider("Grid Resolution", min_value=0.05, max_value=0.5, value=0.1, step=0.05)
# Cache grid in session
if 'grid_res' not in st.session_state or st.session_state.grid_res != resolution:
    st.session_state.grid_res = resolution
    st.session_state.grid = generate_temperature_grid(resolution)
# Load districts once
districts = load_assam_districts()
# Display map
map_obj = create_interactive_map(districts, st.session_state.grid, show_grid, show_districts)
st.subheader("🗺️ Interactive Map")
st_folium(map_obj, width=700, height=500)
# Analytics
st.subheader("📊 Fire Risk Analytics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Average Temperature", f"{districts['temperature'].mean():.1f}°C")
with col2:
    hottest = districts.loc[districts['temperature'].idxmax()]
    st.metric("Hottest District", hottest.district)
st.write("**Risk Distribution**")
for risk, count in districts['fire_risk'].value_counts().items():
    st.write(f"- {risk}: {count}")
high = districts[districts['fire_risk'].isin(['High','Very High'])]
st.write("**High Risk Districts**")
if not high.empty:
    st.dataframe(high[['district','temperature','fire_risk']].sort_values('temperature', ascending=False))
else:
    st.write("No high-risk districts currently.")
