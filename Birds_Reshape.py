from PIL import Image
import os

input_dir = "D:\\CUB_200_2011\\images"
output_dir = "D:\\Birds64"

os.makedirs(output_dir, exist_ok=True)

for class_name in os.listdir(input_dir):
    class_dir = os.path.join(input_dir, class_name)

    for filename in os.listdir(class_dir):
        img_path = os.path.join(class_dir, filename)

        try:
            with Image.open(img_path) as img:
                img_resized = img.resize((64, 64))
                output_path = os.path.join(output_dir, filename)
                img_resized.save(output_path)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

