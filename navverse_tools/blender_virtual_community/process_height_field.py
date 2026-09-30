# external_preprocess.py
import numpy as np
import json
import os
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# city list
cities = ["AMSTERDAM", "AUSTIN", "BALTIMORE", "BARCELONA", "BELGRADE", "BERLIN", "BOISE", "BOSTON", "BRATISLAVA", "BRUSSELS", "BUDAPEST", "CALGARY", "CHARLOTTE", "CHICAGO", "CHRISTCHURCH", "COLUMBUS", "DENVER", "DETROIT", "EL_PASO", "FLORENCE", "FORT_WORTH", "FRANKFURT", "HAMBURG", "HARVARD", "KANSAS_CITY", "LASVEGAS", "LONDON", "LONGISLAND", "MADISON", "MADRID", "MADRID2", "MILAN", "MINNEAPOLIS", "MIT", "MONTREAL", "NY", "ORLANDO", "PARIS", "PHILADELPHIA", "PORTLAND", "ROME", "SANFRANCISCO", "SANFRANCISCO2", "SILICONVALLEY", "STANFORD", "SYDNEY", "TORONTO", "UCLA", "UMASS", "WHITEHOUSE", "YALE", "ZURICH"]

# cities = ["CHARLOTTE"]
DATA_DIR = "D:/Desktop/ViCo"

# main function: loop through all cities
if __name__ == "__main__":
    # create statistics output file
    stats_output_file = f"{DATA_DIR}/generated/city_objects_statistics.txt"
    
    with open(stats_output_file, "w", encoding="utf-8") as stats_file:
        stats_file.write("=== City objects statistics report ===\n")
        stats_file.write(f"Processing time: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        total_all_cities = 0
        total_amenities_all = 0
        total_natural_all = 0
        
        for city in cities:
            print(f"\nProcessing city: {city}")
            
            # file paths
            base_path = f"{DATA_DIR}/{city}"
            height_file = f"{base_path}/height_field.npz"
            amenities_file = f"{base_path}/objects/amenities.json"
            natural_file = f"{base_path}/objects/natural.json"
            output_amenities_file = f"{base_path}/objects/amenities_with_z.json"
            output_natural_file = f"{base_path}/objects/natural_with_z.json"
            
            # check if files exist
            if not os.path.exists(amenities_file) or not os.path.exists(natural_file):
                stats_file.write(f"=== {city} ===\n")
                stats_file.write("Files do not exist, skipping processing\n\n")
                continue
            
            # read JSON
            with open(amenities_file, "r") as f:
                amenities_data = json.load(f)
            with open(natural_file, "r") as f:
                natural_data = json.load(f)
            
            # count various categories
            amenities_count = len(amenities_data["objects"])
            natural_count = len(natural_data["objects"])
            total_count = amenities_count + natural_count
            
            # count amenities categories
            amenities_categories = {}
            for obj in amenities_data["objects"]:
                # add path prefix (only when asset_path starts with "retrieved")
                if "asset_path" in obj and obj["asset_path"].startswith("retrieved"):
                    obj["asset_path"] = f"{DATA_DIR}/objects/outdoor_objects/" + obj["asset_path"]
                
                # count categories
                obj_type = obj.get("tags", {}).get("name", "unknown")
                amenities_categories[obj_type] = amenities_categories.get(obj_type, 0) + 1
            
            # count natural categories
            natural_categories = {}
            for obj in natural_data["objects"]:
                # add path prefix (only when asset_path starts with "retrieved")
                if "asset_path" in obj and obj["asset_path"].startswith("retrieved"):
                    obj["asset_path"] = f"{DATA_DIR}/objects/outdoor_objects/" + obj["asset_path"]
                
                # count categories
                obj_type = obj.get("tags", {}).get("name", "unknown")
                natural_categories[obj_type] = natural_categories.get(obj_type, 0) + 1
            
            # check if there is a height field file, if there is then add z coordinates
            if os.path.exists(height_file):
                # read height field
                height_field = np.load(height_file)
                xs = height_field["plane_coord"][..., 0]
                ys = height_field["plane_coord"][..., 1] * -1  # same as you originally
                zs = height_field["terrain_alt"]

                # create interpolator
                interp_linear = LinearNDInterpolator(np.stack([xs, ys], axis=-1), zs)
                interp_nearest = NearestNDInterpolator(np.stack([xs, ys], axis=-1), zs)

                def get_height(x, y):
                    z = interp_linear(x, y)
                    if np.isnan(z):
                        z = interp_nearest(x, y)
                    return float(z)

                # add z coordinates to each object
                for obj in amenities_data["objects"]:
                    loc = obj.get("location", [0, 0, 0])
                    x, y = loc[0], loc[1]
                    z = get_height(x, y)
                    obj["location"].append(z)
                    
                for obj in natural_data["objects"]:
                    loc = obj.get("location", [0, 0, 0])
                    x, y = loc[0], loc[1]
                    z = get_height(x, y)
                    obj["location"].append(z)
            
            # save JSON file with z coordinates and path prefix
            with open(output_amenities_file, "w", encoding="utf-8") as f:
                json.dump(amenities_data, f, indent=2, ensure_ascii=False)
            with open(output_natural_file, "w", encoding="utf-8") as f:
                json.dump(natural_data, f, indent=2, ensure_ascii=False)
            
            # write statistics file
            stats_file.write(f"=== {city} ===\n")
            stats_file.write(f"Total objects count: {total_count}\n")
            stats_file.write(f"Amenities objects count: {amenities_count}\n")
            stats_file.write(f"Natural objects count: {natural_count}\n")
            
            stats_file.write(f"\nAmenities categories distribution:\n")
            for category, count in sorted(amenities_categories.items()):
                stats_file.write(f"  {category}: {count}\n")
            
            stats_file.write(f"\nNatural categories distribution:\n")
            for category, count in sorted(natural_categories.items()):
                stats_file.write(f"  {category}: {count}\n")
            
            stats_file.write("\n" + "="*50 + "\n\n")
            
            # add up total
            total_all_cities += total_count
            total_amenities_all += amenities_count
            total_natural_all += natural_count
            
            # output to console
            if os.path.exists(height_file):
                print(f"{city} height has been calculated, statistics have been written to file, JSON file has been saved")
            else:
                print(f"{city} statistics have been written to file, JSON file has been saved (no height field file, skipping z coordinate calculation)")
        
        # write total statistics
        stats_file.write("=== Total statistics ===\n")
        stats_file.write(f"Total objects count: {total_all_cities}\n")
        stats_file.write(f"Total Amenities count: {total_amenities_all}\n")
        stats_file.write(f"Total Natural count: {total_natural_all}\n")
        stats_file.write(f"Total cities count: {len(cities)}\n")
    
    print(f"\nAll cities statistics have been saved to: {stats_output_file}")
    print("All cities processing completed!")
