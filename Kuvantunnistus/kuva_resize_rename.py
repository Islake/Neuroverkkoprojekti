import os
from PIL import Image


def resize_and_rename_images(input_folder, output_folder, name, new_width=224, new_height=224):
    prefix = name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for index, file in enumerate(files, start=1):
        img_path = os.path.join(input_folder, file)
        try:
            with Image.open(img_path) as img:
                img = img.resize((new_width, new_height), Image.LANCZOS)

                new_filename = f"{prefix}.{index:02d}.jpg"
                new_path = os.path.join(output_folder, new_filename)
                img.save(new_path, "JPEG")
                print(f"Processed and saved: {new_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


names = ["haarukka", "lusikka", "veitsi"]
for name in names:
    resize_and_rename_images(f"Kuvantunnistus/aterimet/{name}", f"Kuvantunnistus_omilla_kuvilla/kuvantunnistus_small", name)



