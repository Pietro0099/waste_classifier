from pathlib import Path
import imghdr

data_dir = "raw_data/train/plastic"  # Path to directory to check
image_extensions = [".png", ".jpg"]  # Add there all your images file extensions
status = True

img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
for filepath in Path(data_dir).rglob("*"):
	if filepath.suffix.lower() in image_extensions:
		img_type = imghdr.what(filepath)
		if img_type is None:
			print(f"{filepath} is not an image")
			status = False
		elif img_type not in img_type_accepted_by_tf:
			print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
			status = False
if status == True:
	print(f"Status in {data_dir} OK" )
