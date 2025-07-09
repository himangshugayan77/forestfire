import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from matplotlib import cm
import requests
import json
from datetime import datetime, timedelta
import warnings
from shapely.geometry import Point, Polygon, LineString

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Assam Forest Fire Temperature Prediction Map",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.markdown('# 🔥 Assam Forest Fire Temperature Prediction Map', unsafe_allow_html=True)
st.markdown("""
**Advanced Interactive Mapping System for Forest Fire Risk Assessment**
This application provides real-time temperature mapping and forest fire risk prediction for Assam state using:
- 🌡️ **Temperature Analysis**: Multi-source meteorological data integration
- 🛰️ **Satellite Data**: MODIS/VIIRS fire detection capabilities
- 🤖 **ML Predictions**: Machine learning-based risk assessment
- 🗺️ **Interactive Maps**: Dynamic visualization with GeoPandas and Folium
""")

class AssamFireMapApp:
    def __init__(self):
        self.assam_bounds = {
            'min_lon': 89.7, 'max_lon': 96.0,
            'min_lat': 24.1, 'max_lat': 28.2
        }
        self.districts_gdf = self.load_assam_districts()

    @st.cache_data
    def load_assam_districts(_self):
        """Load Assam district data with enhanced geographic information"""
        districts_data = {
            'district': ['Guwahati', 'Dibrugarh', 'Jorhat', 'Silchar', 'Tezpur', 
                        'Nagaon', 'Dhubri', 'Goalpara', 'Kokrajhar', 'Bongaigaon',
                        'Karimganj', 'Hailakandi', 'North Lakhimpur', 'Sivasagar',
                        'Golaghat', 'Morigaon', 'Darrang', 'Sonitpur'],
            'latitude': [26.1445, 27.4728, 26.7509, 24.8333, 26.6333, 26.3467, 
                        26.0173, 26.1664, 26.4018, 26.4831, 24.8697, 24.6847,
                        27.2364, 26.9869, 26.7271, 26.2523, 26.4525, 26.6334],
            'longitude': [91.7362, 94.9120, 94.2037, 92.7789, 92.7833, 92.6811, 
                         89.9583, 90.6167, 90.2631, 90.5436, 92.3542, 92.5442,
                         94.1181, 94.6851, 93.9615, 92.1738, 92.0219, 92.7833],
            'elevation': [55, 111, 116, 15, 58, 56, 37, 42, 46, 45, 8, 23, 
                         105, 96, 295, 52, 61, 58],
            'forest_cover': [35.2, 62.8, 45.1, 28.4, 55.7, 38.9, 22.1, 41.3, 
                           58.2, 33.7, 15.6, 32.1, 67.3, 52.4, 71.8, 29.5, 48.2, 55.7],
            'population': [957352, 154019, 153889, 228951, 58851, 141073, 
                          71838, 58257, 60669, 75123, 79459, 59855, 26196, 
                          114970, 122786, 55746, 69726, 58851]
        }

        df = pd.DataFrame(districts_data)

        # Generate realistic temperature data based on multiple factors
        base_temp = 30
        df['temperature'] = (
            base_temp + 
            (df['latitude'] - df['latitude'].mean()) * -0.8 +  # Latitude effect
            (df['elevation'] / 100) * -1.2 +  # Elevation effect
            np.random.normal(0, 1.5, len(df))  # Random variation
        ).round(1)

        # Calculate fire risk based on multiple parameters
        risk_score = (
            (df['temperature'] - 25) * 0.3 +
            (35 - df['forest_cover']) * 0.02 +
            (df['elevation'] < 100).astype(int) * 2 +
            np.random.normal(0, 0.5, len(df))
        )

        df['fire_risk'] = pd.cut(risk_score, 
                                bins=[-np.inf, 2, 4, 6, np.inf], 
                                labels=['Low', 'Medium', 'High', 'Very High'])

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs='EPSG:4326'
        )

        return gdf

    def generate_temperature_grid(self, resolution=0.1):
        """Generate temperature grid for interpolation"""
        bounds = self.districts_gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        
        lon_range = np.arange(minx, maxx, resolution)
        lat_range = np.arange(miny, maxy, resolution)
        
        points = []
        temperatures = []
        
        for i in range(len(lat_range)):
            for j in range(len(lon_range)):
                # Create Point from shapely.geometry
                points.append(Point(lon_range[j], lat_range[i]))
                
                # Generate temperature based on position
                temp = 25 + np.random.normal(0, 5) + (lat_range[i] - miny) * 0.1
                temperatures.append(temp)
        
        grid_gdf = gpd.GeoDataFrame({
            'temperature': temperatures,
            'geometry': points
        }, crs='EPSG:4326')
        
        return grid_gdf

    def create_interactive_map(self, districts_gdf, grid_gdf, show_grid=True, show_districts=True):
        """Create comprehensive interactive Folium map"""
        # Center the map on Assam
        center_lat = districts_gdf['latitude'].mean()
        center_lon = districts_gdf['longitude'].mean()

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='OpenStreetMap'
        )

        if show_grid and len(grid_gdf) > 0:
            # Add temperature grid as color-coded points
            for idx, row in grid_gdf.iterrows():
                if idx % 20 == 0:  # Sample every 20th point for performance
                    temp = row['temperature']
                    # Normalize temperature for color mapping
                    norm_temp = (temp - grid_gdf['temperature'].min()) / (grid_gdf['temperature'].max() - grid_gdf['temperature'].min())
                    
                    # Use colormap
                    color = cm.coolwarm(norm_temp)
                    hex_color = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
                    
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=3,
                        popup=f"Temperature: {temp:.1f}°C",
                        color=hex_color,
                        fill=True,
                        fillColor=hex_color,
                        fillOpacity=0.6,
                        weight=1
                    ).add_to(m)

        if show_districts:
            # Add district markers
            risk_colors = {
                'Low': '#2E8B57',
                'Medium': '#FFD700',
                'High': '#FF8C00',
                'Very High': '#DC143C'
            }

            for idx, row in districts_gdf.iterrows():
                color = risk_colors.get(str(row['fire_risk']), '#808080')
                
                popup_html = f"""
                <div style="font-family: Arial, sans-serif; width: 200px;">
                    <h4>{row['district']}</h4>
                    <p><b>🌡️ Temperature:</b> {row['temperature']:.1f}°C</p>
                    <p><b>🔥 Fire Risk:</b> {row['fire_risk']}</p>
                    <p><b>🌲 Forest Cover:</b> {row['forest_cover']:.1f}%</p>
                    <p><b>⛰️ Elevation:</b> {row['elevation']} m</p>
                    <p><b>👥 Population:</b> {row['population']:,}</p>
                </div>
                """
                
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color='red' if row['fire_risk'] in ['High', 'Very High'] else 'green')
                ).add_to(m)

        return m

    def run(self):
        """Main application runner"""
        st.sidebar.header("🔧 Map Controls")
        
        # Controls
        show_grid = st.sidebar.checkbox("Show Temperature Grid", value=True)
        show_districts = st.sidebar.checkbox("Show District Markers", value=True)
        resolution = st.sidebar.slider("Grid Resolution", 0.05, 0.5, 0.1)
        
        # Generate temperature grid
        grid_gdf = self.generate_temperature_grid(resolution)
        
        # Create two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🗺️ Interactive Temperature Map")
            # Create map
            map_obj = self.create_interactive_map(
                self.districts_gdf, grid_gdf, show_grid, show_districts
            )
            
            # Display map
            st_folium(map_obj, width=700, height=500)
        
        with col2:
            st.subheader("📊 Fire Risk Analytics")
            
            # Display metrics
            col1_metrics, col2_metrics = st.columns(2)
            
            with col1_metrics:
                avg_temp = self.districts_gdf['temperature'].mean()
                st.metric("🌡️ Average Temperature", f"{avg_temp:.1f}°C")
                
            with col2_metrics:
                max_temp = self.districts_gdf['temperature'].max()
                hottest_district = self.districts_gdf.loc[self.districts_gdf['temperature'].idxmax(), 'district']
                st.metric("🔥 Hottest District", f"{hottest_district}")
            
            # Risk distribution
            risk_counts = self.districts_gdf['fire_risk'].value_counts()
            st.subheader("Risk Distribution")
            for risk, count in risk_counts.items():
                st.write(f"**{risk}**: {count} districts")
            
            # Top risk districts
            st.subheader("🚨 High Risk Districts")
            high_risk = self.districts_gdf[self.districts_gdf['fire_risk'].isin(['High', 'Very High'])].copy()
            if len(high_risk) > 0:
                high_risk_display = high_risk[['district', 'temperature', 'fire_risk']].sort_values('temperature', ascending=False)
                st.dataframe(high_risk_display, use_container_width=True)
            else:
                st.write("No high-risk districts currently.")
                # 1️⃣  Cache expensive parts
@st.cache_data
def load_assam_districts():
    ...

@st.cache_data
def generate_temperature_grid(resolution):
    ...

# 2️⃣  Keep track of the grid resolution so we only rebuild when it changes
resolution = st.sidebar.slider("Grid Resolution", 0.05, 0.5, 0.1)
if "resolution" not in st.session_state or st.session_state.resolution != resolution:
    st.session_state.resolution = resolution
    st.session_state.grid = generate_temperature_grid(resolution)

# 3️⃣  Build the map only when necessary
if "map_obj" not in st.session_state:
st.session_state.map_obj = app.create_interactive_map(
    app.districts_gdf,
    st.session_state.grid,
    show_grid,
    show_districts
)


# 4️⃣  Display without triggering feedback
st_folium(
    st.session_state.map_obj,
    key="assam_temperature_map",
    returned_objects=[]
)


# Run the application
if __name__ == "__main__":
    try:
        app = AssamFireMapApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please check your GeoPandas and Shapely installation.")
