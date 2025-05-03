CREATE TABLE `ship_ais` (
  `id` varchar(32) ,
  `mmsi` varchar(32)  NOT NULL,
  `name` varchar(128) ,
  `latitude` double,
  `longitude` double,
  `sog` double,
  `course` double,
  `heading` double,
  `length` double,
  `draft` double,
  `type` varchar(32) ,
  `tonnage` double,
  `ptms_destination` varchar(32) ,
  `last_updated` datetime,
  `series_id` int,
  `eta` datetime,
  `point` point /*!80003 SRID 4326 */,
  PRIMARY KEY (`mmsi`,`last_updated`)
)

CREATE TABLE `ship_ais_quarter` (
  `id` varchar(32) ,
  `mmsi` varchar(32),
  `name` varchar(128) ,
  `latitude` double,
  `longitude` double,
  `sog` double,
  `course` double,
  `heading` double,
  `length` double,
  `draft` double,
  `type` varchar(32) ,
  `tonnage` double,
  `ptms_destination` varchar(32) ,
  `last_updated` datetime NOT NULL,
  `series_id` bigint,
  `eta` datetime,
  `point` point /*!80003 SRID 4326 */,
  PRIMARY KEY (`mmsi`,`last_updated`),
  KEY `mmsi_index` (`mmsi`),
  KEY `time_index` (`last_updated`) USING BTREE
)

CREATE TABLE `shp_data` (
  `object_type` varchar(255),
  `object_name` varchar(255),
  `geom` geometry NOT NULL , -- polygon or point
  `attributes` json
)

CREATE TABLE `warn_single` (
  `ship1` varchar(32),
  `ship2` varchar(32),E
  `warn_flag` bigint,
  `cpa1_lat` double,
  `cpa2_lat` double,
  `cpa1_lon` double,
  `cpa2_lon` double,
  `dcpa` double,
  `tcpa` double,
  `range` double,
  `dt` datetime NOT NULL,
  PRIMARY KEY (`ship1`,`ship2`,`dt`)
)

-- Ship AIS information table
-- - `ship_ais`: Stores **real-time** ship position data.
-- - `ship_ais_quarter`: Stores ship data for the **past 15 minutes** (10s intervals).
-- - ship_ais.point/ship_ais_quarter.point is already a geometry Point object in the correct format. Do not use POINT(s.longitude, s.latitude), just use ship_ais.point or ship_ais_quarter.point
-- - 
-- shp_data table
-- - `object_type`: **Geofence type** (`"anchorage"`, `"fairway"`, `"pilot_boarding_ground"`, `"strait"`) 
-- - `object_name`: **Geofence name** (may have multiple areas)  
-- - `geom`: **Spatial data type**
-- - `POLYGON`: `fairway`, `anchorage`, `strait` 
-- - `POINT`: `pilot_boarding_ground`
--  warn_sigle table
-- - `ship1` / `ship2`: **MMSI of ships**
-- - `dt`: **Report timestamp**