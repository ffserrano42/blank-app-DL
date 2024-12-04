import os
import re

folder_path = 'data/cropped_labels_corrected'

pattern = re.compile(r'^0\.0 0\.[0-9]{6} 0\.[0-9]{6} 0\.0{6} 0\.0{6}$')

def clean_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and filename.endswith('.txt'):
            with open(file_path, 'r') as file:
                lines = file.readlines()
            filtered_lines = [line for line in lines if not pattern.match(line.strip())]

            if len(filtered_lines) != len(lines):
                with open(file_path, 'w') as file:
                    file.writelines(filtered_lines)
                print(f"Líneas coincidentes eliminadas en: {filename}")
            else:
                print(f"No se encontraron coincidencias en: {filename}")

clean_files_in_folder(folder_path)

