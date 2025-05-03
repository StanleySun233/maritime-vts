import os
from datetime import datetime, timedelta

import folium
import matplotlib.pyplot as plt
import pandas as pd
import pymysql
from folium.plugins import TimestampedGeoJson
from matplotlib.colors import to_hex

import util


def query_ship_data(mmsi1, mmsi2, dt, mysql_db):
    # 将dt解析为datetime对象
    dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')

    # 计算时间范围
    start_time = dt - timedelta(minutes=5)
    end_time = dt + timedelta(minutes=5)

    # 将时间范围格式化为字符串
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')

    # 定义SQL查询语句
    query = f"""
        SELECT *
        FROM ship_ais_hive
        WHERE (mmsi = {mmsi1} OR mmsi = {mmsi2})
        AND last_updated BETWEEN '{start_time_str}' AND '{end_time_str}'
    """
    return pd.read_sql(query, mysql_db)


import math


def rotate_polygon(lists, angle):
    # 将角度转换为弧度
    rad = math.radians(90 + angle)

    # 计算旋转中心（即多边形重心）
    lon_sum, lat_sum = 0, 0
    for lon, lat in lists:
        lon_sum += lon
        lat_sum += lat
    center_lon, center_lat = lon_sum / len(lists), lat_sum / len(lists)

    # 定义旋转公式：相对中心点旋转
    def rotate_point(lon, lat):
        # 将点转换为相对中心坐标
        relative_lon = lon - center_lon
        relative_lat = lat - center_lat

        # 应用旋转矩阵
        rotated_lon = relative_lon * math.cos(rad) - relative_lat * math.sin(rad)
        rotated_lat = relative_lon * math.sin(rad) + relative_lat * math.cos(rad)

        # 转换回原坐标系
        return [rotated_lon + center_lon, rotated_lat + center_lat]

    # 对每个点进行旋转
    rotated_list = [rotate_point(lon, lat) for lon, lat in lists]

    return rotated_list


def save_html_page(df, tcpa, dcpa, output_dir="./webhtml"):
    df['last_updated'] = pd.to_datetime(df['last_updated'])

    # 获取不同船舶的 MMSI 列表和为每艘船舶分配颜色
    unique_mmsis = df['mmsi'].unique()
    colors = plt.cm.get_cmap('tab10', len(unique_mmsis))
    mmsi_color_map = {mmsi: to_hex(colors(i)) for i, mmsi in enumerate(unique_mmsis)}

    # 创建 GeoJSON 格式数据
    features = []
    for _, row in df.iterrows():
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [row['longitude'], row['latitude']]
            },
            'properties': {
                'time': row['last_updated'].isoformat(),
                'popup': f"MMSI: {row['mmsi']}",
                'style': {'color': mmsi_color_map[row['mmsi']]},
                'icon': 'circle',
                'iconstyle': {
                    'fillColor': mmsi_color_map[row['mmsi']],
                    'fillOpacity': 0.6,
                    'stroke': 'true',
                    'radius': 5
                }

            }
        }
        features.append(feature)

    width_half = 0.0005
    height_half = 0.0003

    for _, row in df.iterrows():
        center_lon, center_lat = row['longitude'], row['latitude']

        # Calculate the rectangle's corner points
        corners = [
            [center_lon - width_half, center_lat - height_half],  # Bottom-left corner
            [center_lon + width_half, center_lat - height_half],  # Bottom-right corner
            [center_lon + width_half, center_lat + height_half],  # Top-right corner
            [center_lon - width_half, center_lat + height_half],  # Top-left corner
        ]
        corners = rotate_polygon(corners, row['course'])
        corners.append(corners[0])  # Close the polygon by adding the first point again

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [corners]  # GeoJSON expects a nested list of coordinates for polygons
            },
            'properties': {
                'time': row['last_updated'].isoformat(),
                'popup': f"MMSI: {row['mmsi']}",
                'style': {'color': mmsi_color_map[row['mmsi']]},
                'icon': 'rectangle',
                'iconstyle': {
                    'fillColor': mmsi_color_map[row['mmsi']],
                    'fillOpacity': 0.6,
                    'stroke': 'true'
                }
            }
        }
        features.append(feature)

    # 构建时间序列 GeoJSON 数据
    geojson_data = {
        'type': 'FeatureCollection',
        'features': features
    }
    # 创建 Folium 地图
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=15)

    # 添加 TimestampedGeoJson 图层
    TimestampedGeoJson(
        geojson_data,
        period='PT10S',  # 每步5分钟
        add_last_point=True,
        auto_play=True,
        loop=False,
        max_speed=1,
        loop_button=True,
        time_slider_drag_update=True
    ).add_to(m)

    # 保存HTML文件
    timestamp = str(int(datetime.now().timestamp()))
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'ship_encounter_{timestamp}.html')
    m.save(file_path)
    print(f"HTML page saved to {file_path}")


def web_data_table():
    pass


def web_collision_analysis(df):
    """
    生成一个html模板，用于python可视化展示船舶碰撞风险检测。
    输入df包含：mmsi,name,latitude,longitude
    """

    pass


if __name__ == "__main__":
    cfg = util.load_config("../config.yaml")
    mysql_db = pymysql.Connection(host=cfg["mysql"]['host'],
                                  user=cfg["mysql"]['user'],
                                  port=int(cfg["mysql"]['port']),
                                  password=cfg["mysql"]['password'],
                                  database=cfg["mysql"]['database'], )
    tcpa = 10
    dcpa = 0.05
    df = query_ship_data("636014693", "574013228", "2022-12-07 16:44:30", mysql_db)
    save_html_page(df, 10, 0.05, "../webhtml")
