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


def process_shapefiles(shp_files, table_name="shp_data"):
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

    for shp_file in shp_files:
        shp_name = format_shp_name(shp_file).split("/")[-1].split(".")[0]
        gdf = gpd.read_file(shp_file)
        gdf['geometry'] = gdf['geometry'].apply(swap_coordinates)
        print(gdf)
        # Add name, sid, and OBJNAM fields
        gdf["object_type"] = shp_name
        gdf["sid"] = [_ + 1 for _ in range(len(gdf))]
        if 'Strait' not in shp_file:
            gdf["object_name"] = gdf.get("OBJNAM", "Unknown")  # Ensure OBJNAM exists
        else:
            gdf["object_name"] = gdf.get("layer", "Unknown")
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
