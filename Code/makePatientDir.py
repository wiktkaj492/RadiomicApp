def split_into_folders(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)


        if os.path.isfile(file_path) and file.lower().endswith('.png'):
            patient_number = file.split('_')[1].zfill(3)

            patient_folder = os.path.join(output_folder, patient_number)
            if not os.path.exists(patient_folder):
                os.makedirs(patient_folder)

            shutil.copy(file_path, patient_folder)

if __name__ == "__main__":

    input_folder = '/path/to/input/folder'
    output_folder = '/path/to/output/folder'
    split_into_folders(input_folder, output_folder)
