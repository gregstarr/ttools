import apexpy
import os
import glob

from ttools import io, utils, convert, config, old, satellite

swarm_files = old.filter_swarm_files(glob.glob(os.path.join(config.swarm_dir, "extd_efi_lp", "SW_EXTD_EFI*.cdf")))
for file in swarm_files:
    name = utils.no_ext_fn(file)
    coords_fn = os.path.join(config.swarm_coords_dir, f"{name}_coords.h5")

    swarm_data = old.open_swarm_file(file)
    lat, lon = satellite.fix_latlon(swarm_data['Latitude'], swarm_data['Longitude'])

    converter = apexpy.Apex(utils.datetime64_to_datetime(swarm_data['Timestamp'][0]))
    apex_lat, apex_lon = converter.convert(lat, lon, 'geo', 'apex', swarm_data['Height'])
    qd_lat, qd_lon = converter.convert(lat, lon, 'geo', 'qd', swarm_data['Height'])
    mlt = convert.mlon_to_mlt_array(apex_lon, swarm_data['Timestamp'], converter)

    io.write_h5(coords_fn, apex_lat=apex_lat, apex_lon=apex_lon, qd_lat=qd_lat, qd_lon=qd_lon, mlt=mlt, lat=lat, lon=lon)