import os
import re

folder_path = 'data/cropped_labels_corrected'

def clean_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and filename.endswith('.txt'):
            with open(file_path, 'r') as file:
                lines = file.readlines()

            unique_lines = set()
            for line in lines:
                unique_lines.add(line)

            if len(unique_lines) != len(lines):
                with open(file_path, 'w') as file:
                    file.writelines(unique_lines)
                print(f"Archivo procesado y guardado sin duplicados ni coincidencias: {filename}")
            else:
                print(f"No se encontraron duplicados ni coincidencias en: {filename}")

clean_files_in_folder(folder_path)
