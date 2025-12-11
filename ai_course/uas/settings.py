from ultralytics import settings

# Update a setting
settings.update({"runs_dir": "...\\runs"})
settings.update({"datasets_dir": "...\\dataset"})
settings.update({"weights_dir": "...\\weights"})

# Reset settings to default values
# settings.reset()