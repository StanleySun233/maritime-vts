import glob
import os
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pymysql
import yaml


def load_config(path="./config.yaml"):
    """
    Load database configuration from config.yaml.
    """
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def load_mysql_connection():
    cfg = load_config()
    conn = pymysql.connect(
        host=cfg['mysql']['host'],
        user=cfg['mysql']['user'],
        password=cfg['mysql']['password'],
        database=cfg['mysql']['database'],
        port=cfg['mysql']['port'],
    )
    return conn


def format_shp_name(shp_file):
    # Remove the file extension and format the name to lowercase with underscores
    base_name = shp_file.rsplit('.', 1)[0]
    formatted_name = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', base_name).lower()
    return formatted_name


def load_gdf(folder_path="./data/wk/"):
    # 获取所有 .gdf 文件
    gdf_files = glob.glob(os.path.join(folder_path, "*.shp"))

    # 读取所有 GDF 文件并合并
    gdfs = []
    for file in gdf_files:
        gdf = gpd.read_file(file)
        shp_name = format_shp_name(file).split("/")[-1].split(".")[0].replace("wk\\", "")
        # 根据文件名条件设置 object_name 列
        if 'Strait' not in file:
            gdf["object_name"] = gdf["OBJNAM"] if "OBJNAM" in gdf.columns else "Unknown"
        else:
            gdf["object_name"] = gdf["layer"] if "layer" in gdf.columns else "Unknown"
        gdf["object_type"] = shp_name
        gdfs.append(gdf)
    combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    # 检查并统一 CRS（坐标参考系统）
    if not combined_gdf.crs:
        combined_gdf.set_crs(gdfs[0].crs, inplace=True)
    else:
        combined_gdf.to_crs(gdfs[0].crs, inplace=True)
    return combined_gdf


def get_sql(_sql):
    conn = load_mysql_connection()
    df = pd.read_sql(_sql, conn)
    return df


if __name__ == '__main__':
    gdf = load_gdf()
    # print(gdf)
    gdf = gdf[gdf['object_type'] == 'fairway']
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.6)

    # 添加标题
    plt.title("Fairway Areas")

    # 显示地图
    plt.show()
