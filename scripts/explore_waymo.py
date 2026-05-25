import pandas as pd
from pathlib import Path

#Base Project path

BASE_PATH = Path(r"C:\Users\Anthony Melgar\Desktop\Portafolio\WaymoProject\parquet")


#Daset Folders

camera_box_path = BASE_PATH / "camera_box"
lidar_box_path = BASE_PATH / "lidar_box"
stats_path = BASE_PATH / "stats"

#load parquet file from camera_box

camera_files = list(camera_box_path.glob("*.parquet"))

print(f"Found {len(camera_files)} camera_box files")


#Read first parquet

df_camera = pd.read_parquet(camera_files[0])

print("\n -------- CAMERA BOX DATA -----------")
print(df_camera.head())

print("\n -------- COLUMNS -----------")
print(df_camera.columns)

print("\n -------- DATA INFO -----------")
print(df_camera.info())

print("\n -------- UNIQUE CAMERA IDS -----------")
print(df_camera["key.camera_name"].unique())

print("\n -------- UNIQUE OBJECT TYPES -----------")
print(df_camera["[CameraBoxComponent].type"].unique())

print("\n -------- OBJECT TYPE COUNTS -----------")
print(df_camera["[CameraBoxComponent].type"].value_counts())


#Create cleaner dataframe

df_clean = df_camera.rename(columns={
    "key.segment_context_name": "segment_name",
    "key.frame_timestamp_micros": "timestamp",
    "key.camera_name": "camera_id",
    "key.camera_object_id": "object_id",
    "[CameraBoxComponent].box.center.x": "center_x",
    "[CameraBoxComponent].box.center.y": "center_y",
    "[CameraBoxComponent].box.size.x": "box_width",
    "[CameraBoxComponent].box.size.y": "box_height",
    "[CameraBoxComponent].type": "object_type",
    "[CameraBoxComponent].difficulty_level.detection": "detection_difficulty",
    "[CameraBoxComponent].difficulty_level.tracking": "tracking_difficulty"
})

print("\n -------- CLEAN DATAFRAME --------")
print(df_clean.head())


# Camera ID mapping

camera_map = {
    1: "CAM_1",
    2: "CAM_2",
    3: "CAM_3",
    4: "CAM_4",
    5: "CAM_5"
}

# Object type mapping

object_map = {
    1: "TYPE_1",
    2: "TYPE_2",
    3: "TYPE_3",
    4: "TYPE_4"
}

# Apply mappings

df_clean["camera_name"] = df_clean["camera_id"].map(camera_map)

df_clean["object_name"] = df_clean["object_type"].map(object_map)

print("\n -------- MAPPED DATAFRAME --------")
print(
    df_clean[
        ["camera_id", "camera_name", "object_type", "object_name"]
    ].head()
)

# Create bounding box area

df_clean["box_area"] = (
    df_clean["box_width"] * df_clean["box_height"]
)

# Detection difficulty flag

df_clean["is_difficult_detection"] = (
    df_clean["detection_difficulty"].notnull()
)

# Tracking difficulty flag

df_clean["is_difficult_tracking"] = (
    df_clean["tracking_difficulty"].notnull()
)

print("\n -------- NEW FEATURES --------")
print(
    df_clean[
        [
            "box_width",
            "box_height",
            "box_area",
            "is_difficult_detection",
            "is_difficult_tracking"
        ]
    ].head()
)

# Combine all parquet files

all_dataframes = []

for file in camera_files:
    
    temp_df = pd.read_parquet(file)
    
    temp_df = temp_df.rename(columns={
        "key.segment_context_name": "segment_name",
        "key.frame_timestamp_micros": "timestamp",
        "key.camera_name": "camera_id",
        "key.camera_object_id": "object_id",
        "[CameraBoxComponent].box.center.x": "center_x",
        "[CameraBoxComponent].box.center.y": "center_y",
        "[CameraBoxComponent].box.size.x": "box_width",
        "[CameraBoxComponent].box.size.y": "box_height",
        "[CameraBoxComponent].type": "object_type",
        "[CameraBoxComponent].difficulty_level.detection": "detection_difficulty",
        "[CameraBoxComponent].difficulty_level.tracking": "tracking_difficulty"
    })

    all_dataframes.append(temp_df)

# Merge everything

master_df = pd.concat(all_dataframes, ignore_index=True)

print("\n -------- MASTER DATASET --------")
print(master_df.shape)

print("\n -------- MASTER DATA SAMPLE --------")
print(master_df.head())

# Apply mappings to master dataset

master_df["camera_name"] = master_df["camera_id"].map(camera_map)

master_df["object_name"] = master_df["object_type"].map(object_map)

# Create box area

master_df["box_area"] = (
    master_df["box_width"] * master_df["box_height"]
)

# Difficult detection flag

master_df["is_difficult_detection"] = (
    master_df["detection_difficulty"].notnull()
)

# Difficult tracking flag

master_df["is_difficult_tracking"] = (
    master_df["tracking_difficulty"].notnull()
)

print("\n -------- FINAL MASTER DATASET --------")
print(master_df.head())

print("\n -------- MASTER DATASET SHAPE --------")
print(master_df.shape)

# Export final dataset

output_path = r"C:\Users\Anthony Melgar\Desktop\Portafolio\WaymoProject\csv\waymo_master_dataset.csv"

master_df.to_csv(output_path, index=False)

print("\n CSV exported successfully!")
print(output_path)
                    

