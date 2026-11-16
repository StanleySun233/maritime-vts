import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
AVAILABLE = all(importlib.util.find_spec(name) for name in ("geopandas", "shapely", "yaml"))


@unittest.skipUnless(AVAILABLE, "geospatial import dependencies are not installed")
class GeodataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import geopandas
        from shapely.geometry import Point, Polygon

        spec = importlib.util.spec_from_file_location("geodata_loader", PROJECT / "data" / "load_shp_data.py")
        cls.loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.loader)
        cls.geopandas = geopandas
        cls.Point = Point
        cls.Polygon = Polygon

    def test_real_strait_geometry_preserves_holes(self):
        data = self.geopandas.read_file(PROJECT / "data" / "general" / "Strait.shp")
        original = next(geom for geom in data.geometry if len(geom.interiors) > 0)
        converted = self.loader.swap_coordinates(original)
        self.assertEqual(len(original.interiors), 7)
        self.assertEqual(len(converted.interiors), len(original.interiors))
        self.assertAlmostEqual(converted.area, original.area)
        inside_hole = self.Polygon(original.interiors[0]).representative_point()
        swapped_point = self.Point(inside_hole.y, inside_hole.x)
        self.assertFalse(converted.contains(swapped_point))

    def test_bundled_pilot_point_coordinates_are_swapped(self):
        data = self.geopandas.read_file(PROJECT / "data" / "general" / "PilotBoardingGround.shp")
        original = data.geometry.iloc[0]
        converted = self.loader.swap_coordinates(original)
        self.assertEqual((converted.x, converted.y), (original.y, original.x))
        self.assertFalse(converted.has_z)

    def test_three_dimensional_point_retains_elevation(self):
        converted = self.loader.swap_coordinates(self.Point(10, 20, 30))
        self.assertEqual((converted.x, converted.y, converted.z), (20, 10, 30))

    def test_bundled_port_uses_existing_type_and_name(self):
        path = PROJECT / "data" / "port" / "singapore_port.shp"
        original = self.geopandas.read_file(path).iloc[0]
        prepared = self.loader.prepare_shapefile(str(path)).iloc[0]
        self.assertEqual(prepared["object_type"], original["object_typ"])
        self.assertEqual(prepared["object_name"], original["object_nam"])
        self.assertEqual(prepared["object_type"], "port")

    def test_missing_names_are_mysql_null_and_anchorage_type_is_normalized(self):
        from pymysql.converters import escape_item

        path = PROJECT / "data" / "general" / "Ancharage.shp"
        data = self.loader.prepare_shapefile(str(path))
        self.assertEqual(set(data["object_type"]), {"anchorage"})
        missing = [value for value in data["object_name"] if value is None]
        self.assertEqual(len(missing), 8)
        escaped = [escape_item(value, "utf8mb4") for value in data["object_name"]]
        self.assertEqual(escaped.count("NULL"), 8)

    def test_strait_names_come_from_source_layer(self):
        path = PROJECT / "data" / "general" / "Strait.shp"
        original = self.geopandas.read_file(path)
        prepared = self.loader.prepare_shapefile(str(path))
        self.assertEqual(prepared["object_name"].tolist(), original["layer"].tolist())

    def test_default_shapefile_exists_without_database_access(self):
        with patch.object(sys, "argv", ["load_shp_data.py"]), patch.object(self.loader, "process_shapefiles") as process:
            self.loader.main()
        paths = process.call_args.args[0]
        self.assertEqual(paths, [str(PROJECT / "data" / "port" / "singapore_port.shp")])
        self.assertTrue(Path(paths[0]).is_file())
