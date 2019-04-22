# run this in the folder that contains shapefiles, make sure there is a folder called geojson-aois next to it
for f in *.shp; do ogr2ogr -f geojson ../geojson-aois/$f-converted.geojson $f; done
