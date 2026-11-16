import argparse
import re
from pathlib import Path

import geopandas as gpd
import pymysql
import yaml
from shapely import transform


def format_shp_name(shp_file):
    # Remove the file extension and format the name to lowercase with underscores
    base_name = shp_file.rsplit('.', 1)[0]
    formatted_name = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', base_name).lower()
    return formatted_name


def load_config():
    with open(Path(__file__).resolve().parent.parent / "config.yaml", "r", encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
    return cfg["mysql"]


def connect_to_mysql(config):
    return pymysql.connect(
        host=config["host"],
        user=config["user"],
        password=config["password"],
        database=config["database"],
        port=config.get("port", 3306),
    )


def swap_coordinates(geom):
    return transform(geom, lambda coordinates: coordinates[:, [1, 0, 2]], include_z=True)


def prepare_shapefile(shp_file):
    shp_name = format_shp_name(Path(shp_file).name)
    if shp_name == "ancharage":
        shp_name = "anchorage"
    gdf = gpd.read_file(shp_file)
    if gdf.crs is None:
        raise ValueError(f"Missing coordinate reference system: {shp_file}")
    gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = gdf.geometry.apply(swap_coordinates)
    gdf["object_type"] = gdf["object_typ"].fillna(shp_name) if "object_typ" in gdf else shp_name
    name_field = next((field for field in ("object_nam", "OBJNAM", "layer") if field in gdf), None)
    gdf["object_name"] = gdf[name_field].astype(object).where(gdf[name_field].notna(), None) if name_field else None
    gdf["sid"] = range(1, len(gdf) + 1)
    return gdf


def process_shapefiles(shp_files, table_name="shp_data"):
    prepared = [(shp_file, prepare_shapefile(shp_file)) for shp_file in shp_files]
    config = load_config()
    connection = connect_to_mysql(config)
    cursor = connection.cursor()

    # Create table if it doesn't exist
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        object_type VARCHAR(255),
        sid INT,
        object_name VARCHAR(255),
        geom GEOMETRY NOT NULL,
        attributes JSON
    );
    """
    cursor.execute(create_table_query)

    for shp_file, gdf in prepared:
        print(gdf)
        print(gdf["object_name"])
        for _, row in gdf.iterrows():
            geom_wkt = row.geometry.wkt  # Convert geometry to WKT format
            attributes = row.drop("geometry").to_json()  # Convert attributes to JSON

            insert_query = f"""
            INSERT INTO {table_name} (object_type, sid, object_name, geom, attributes)
            VALUES (%s, %s, %s, ST_GeomFromText(%s, 4326), %s)
            """
            cursor.execute(insert_query, (row["object_type"], row["sid"], row["object_name"], geom_wkt, attributes))

        print(f"Data successfully inserted: {shp_file}")

    connection.commit()
    cursor.close()
    connection.close()
    print("All data successfully inserted into the database!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shapefiles", nargs="*", default=[str(Path(__file__).resolve().parent / "port" / "singapore_port.shp")])
    args = parser.parse_args()
    process_shapefiles(args.shapefiles)


if __name__ == "__main__":
    main()
