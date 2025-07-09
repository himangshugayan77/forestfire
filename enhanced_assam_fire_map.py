
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
# CORRECT IMPORT - Add this at the top of your file
from shapely.geometry import Point, Polygon, LineString

# Alternative import for newer Shapely versions
from shapely import Point

# Keep your existing GeoPandas import
import geopandas as gpd

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Assam Forest Fire Temperature Prediction Map",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-legend {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="main-header">🔥 Assam Forest Fire Temperature Prediction Map</p>', unsafe_allow_html=True)

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

from shapely.geometry import Point  # ESSENTIAL IMPORT

@st.cache_data
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
            # FIXED: Use Point from shapely.geometry
            points.append(Point(lon_range[j], lat_range[i]))
            
            # Generate temperature based on position
            temp = 25 + np.random.normal(0, 5) + (lat_range[i] - miny) * 0.1
            temperatures.append(temp)
    
    grid_gdf = gpd.GeoDataFrame({
        'temperature': temperatures,
        'geometry': points
    })
    
    return grid_gdf


    def create_interactive_map(self, districts_gdf, grid_gdf, show_grid=True, show_districts=True):
        """Create comprehensive interactive Folium map"""
        # Center the map on Assam
        center_lat = districts_gdf['latitude'].mean()
        center_lon = districts_gdf['longitude'].mean()

        # Create base map with satellite imagery option
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='OpenStreetMap'
        )

        if show_grid and len(grid_gdf) > 0:
            # Add temperature grid as color-coded points
            for idx, row in grid_gdf.iterrows():
                if idx % 10 == 0:  # Sample every 10th point for performance
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
            # Add district markers with detailed information
            for idx, row in districts_gdf.iterrows():
                # Color based on fire risk
                risk_colors = {
                    'Low': '#2E8B57',      # Sea Green
                    'Medium': '#FFD700',    # Gold
                    'High': '#FF8C00',      # Dark Orange
                    'Very High': '#DC143C'  # Crimson
                }

                color = risk_colors.get(str(row['fire_risk']), '#808080')

                # Create detailed popup
                popup_html = f"""
                <div style="font-family: Arial; width: 250px;">
                    <h4 style="color: {color}; margin-bottom: 10px;">
                        📍 {row['district']}
                    </h4>
                    <hr style="margin: 5px 0;">
                    <p><b>🌡️ Temperature:</b> {row['temperature']:.1f}°C</p>
                    <p><b>🔥 Fire Risk:</b> <span style="color: {color};">{row['fire_risk']}</span></p>
                    <p><b>🌲 Forest Cover:</b> {row['forest_cover']:.1f}%</p>
                    <p><b>⛰️ Elevation:</b> {row['elevation']} m</p>
                    <p><b>👥 Population:</b> {row['population']:,}</p>
                    <p><b>📍 Coordinates:</b> {row['latitude']:.3f}°N, {row['longitude']:.3f}°E</p>
                </div>
                """

                # Add marker
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=8 + (row['temperature'] - districts_gdf['temperature'].min()) * 0.5,
                    popup=folium.Popup(popup_html, max_width=300),
                    color='white',
                    weight=2,
                    fillColor=color,
                    fillOpacity=0.8,
                    tooltip=f"{row['district']}: {row['temperature']:.1f}°C"
                ).add_to(m)

        # Add custom legend HTML
        legend_html = f"""
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 160px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px; border-radius: 5px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2);">
        <h4 style="margin-top: 0; color: #333;">🔥 Fire Risk Legend</h4>
        <p><span style="color:#2E8B57; font-size: 20px;">●</span> Low Risk (&lt; 29°C)</p>
        <p><span style="color:#FFD700; font-size: 20px;">●</span> Medium Risk (29-31°C)</p>
        <p><span style="color:#FF8C00; font-size: 20px;">●</span> High Risk (31-33°C)</p>
        <p><span style="color:#DC143C; font-size: 20px;">●</span> Very High Risk (&gt; 33°C)</p>
        <p style="font-style: italic; color: #666; margin-top: 10px;">
            Grid shows temperature distribution
        </p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        return m

    def create_analytics_dashboard(self, districts_gdf):
        """Create comprehensive analytics dashboard"""
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Temperature Distribution")

            # Temperature histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(districts_gdf['temperature'], bins=15, alpha=0.7, color='orange', edgecolor='black')
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel('Number of Districts')
            ax.set_title('Temperature Distribution Across Assam Districts')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Risk vs Forest Cover scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            risk_colors = {'Low': 'green', 'Medium': 'gold', 'High': 'orange', 'Very High': 'red'}

            for risk in districts_gdf['fire_risk'].unique():
                mask = districts_gdf['fire_risk'] == risk
                ax.scatter(
                    districts_gdf[mask]['forest_cover'], 
                    districts_gdf[mask]['temperature'],
                    c=risk_colors.get(str(risk), 'gray'), 
                    label=risk, 
                    s=60, 
                    alpha=0.7
                )

            ax.set_xlabel('Forest Cover (%)')
            ax.set_ylabel('Temperature (°C)')
            ax.set_title('Temperature vs Forest Cover by Risk Level')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with col2:
            st.subheader("🎯 Risk Analysis")

            # Risk distribution pie chart
            risk_counts = districts_gdf['fire_risk'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = ['#2E8B57', '#FFD700', '#FF8C00', '#DC143C']
            ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                   colors=colors[:len(risk_counts)], startangle=90)
            ax.set_title('Fire Risk Distribution')
            st.pyplot(fig)

            # Top risk districts table
            st.subheader("⚠️ Highest Risk Districts")
            high_risk = districts_gdf[districts_gdf['fire_risk'].isin(['High', 'Very High'])].sort_values('temperature', ascending=False)

            if not high_risk.empty:
                display_df = high_risk[['district', 'temperature', 'fire_risk', 'forest_cover']].head(10)
                display_df.columns = ['District', 'Temperature (°C)', 'Risk Level', 'Forest Cover (%)']
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No high-risk districts found.")

    def show_data_sources_info(self):
        """Display information about data sources and integration"""
        st.subheader("📡 Data Sources & Integration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🛰️ Satellite Data Sources:**
            - **MODIS**: Moderate Resolution Imaging Spectroradiometer
            - **VIIRS**: Visible Infrared Imaging Radiometer Suite
            - **Landsat**: High-resolution land surface data

            **🌡️ Meteorological Data:**
            - **IMD**: Indian Meteorological Department
            - **ERA5**: ECMWF Reanalysis data
            - **Local Weather Stations**: Real-time temperature data
            """)

        with col2:
            st.markdown("""
            **🤖 Machine Learning Models:**
            - **SVM**: Support Vector Machine (90.8% accuracy)
            - **ANN**: Artificial Neural Network (90.3% accuracy)
            - **Random Forest**: Ensemble method for risk prediction

            **📊 Risk Factors:**
            - Temperature, Humidity, Wind Speed
            - Vegetation Index (NDVI)
            - Topographical features
            - Historical fire data
            """)

    def run(self):
        """Main application runner"""
        # Sidebar controls
        st.sidebar.header("🎛️ Map Controls")

        # Map options
        map_type = st.sidebar.selectbox(
            "Select Visualization",
            ["Interactive Map", "Analytics Dashboard", "Both", "Data Sources Info"]
        )

        show_grid = st.sidebar.checkbox("Show Temperature Grid", value=True)
        show_districts = st.sidebar.checkbox("Show District Markers", value=True)

        # Grid resolution
        if show_grid:
            resolution = st.sidebar.slider("Grid Resolution", 0.02, 0.1, 0.05, 0.01)
        else:
            resolution = 0.05

        # Temperature threshold
        temp_threshold = st.sidebar.slider("High Risk Temperature Threshold (°C)", 28, 35, 32)

        # Load data
        with st.spinner("🔄 Loading geospatial data..."):
            districts_gdf = self.load_assam_districts()
            grid_gdf = self.generate_temperature_grid(resolution)

        # Update risk categories based on threshold
        districts_gdf['fire_risk_updated'] = pd.cut(
            districts_gdf['temperature'], 
            bins=[-np.inf, temp_threshold-3, temp_threshold-1, temp_threshold+1, np.inf],
            labels=['Low', 'Medium', 'High', 'Very High']
        )

        # Main metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_temp = districts_gdf['temperature'].mean()
            st.metric("🌡️ Average Temperature", f"{avg_temp:.1f}°C")

        with col2:
            max_temp = districts_gdf['temperature'].max()
            hottest_district = districts_gdf.loc[districts_gdf['temperature'].idxmax(), 'district']
            st.metric("🔥 Hottest District", f"{hottest_district}", f"{max_temp:.1f}°C")

        with col3:
            high_risk_count = len(districts_gdf[districts_gdf['fire_risk_updated'].isin(['High', 'Very High'])])
            st.metric("⚠️ High Risk Districts", high_risk_count)

        with col4:
            avg_forest_cover = districts_gdf['forest_cover'].mean()
            st.metric("🌲 Average Forest Cover", f"{avg_forest_cover:.1f}%")

        # Display content based on selection
        if map_type in ["Interactive Map", "Both"]:
            st.subheader("🗺️ Interactive Temperature & Fire Risk Map")

            # Create and display map
            interactive_map = self.create_interactive_map(
                districts_gdf, grid_gdf, show_grid, show_districts
            )

            map_data = st_folium(interactive_map, width=1200, height=600)

            # Show clicked district info
            if map_data and 'last_object_clicked_popup' in map_data:
                st.info("💡 Click on district markers to view detailed information!")

        if map_type in ["Analytics Dashboard", "Both"]:
            self.create_analytics_dashboard(districts_gdf)

        if map_type == "Data Sources Info":
            self.show_data_sources_info()

        # Data table
        with st.expander("📋 Complete District Data", expanded=False):
            display_columns = ['district', 'latitude', 'longitude', 'temperature', 
                             'fire_risk', 'forest_cover', 'elevation', 'population']
            st.dataframe(districts_gdf[display_columns], use_container_width=True)

        # Footer
        st.markdown("---")
        st.markdown("""
        **🔬 Research Notes:** This application demonstrates forest fire risk mapping techniques based on 
        peer-reviewed research showing temperature as a key predictor. For production use, integrate with 
        real-time meteorological data and validated ML models.

        **📚 Data Sources:** Sample data for demonstration. Recommended sources include NASA FIRMS, 
        Indian Meteorological Department, and Forest Survey of India.
        """)
# Convert geometry for Streamlit display
if 'geometry' in your_gdf.columns:
    display_df = your_gdf.copy()
    display_df['geometry'] = display_df['geometry'].astype(str)
    st.dataframe(display_df)
# More efficient for multiple points
import geopandas as gpd
from shapely.geometry import Point

# Method 1: Using gpd.points_from_xy (recommended)
gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(longitude_array, latitude_array)
)

# Method 2: List comprehension for custom data
points = [Point(lon, lat) for lon, lat in zip(longitude_array, latitude_array)]
gdf = gpd.GeoDataFrame(geometry=points)
@st.cache_data
def load_spatial_data():
    # Load and process your spatial data
    return gdf

# Use the cached function
cached_gdf = load_spatial_data()
try:
    from shapely.geometry import Point
    # Your spatial operations
except ImportError as e:
    st.error(f"Spatial libraries not available: {e}")
    st.stop()
# Run the application
if __name__ == "__main__":
    app = AssamFireMapApp()
    app.run()
