import os

root_dir = 'D:\CUB_200_2011\\text_c10'

output_file = 'D:\output_train_bird_test.txt'
counter=0
with open(output_file, 'w') as f_out:
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            text_dir = os.path.join(root, dir_name)
            for filename in os.listdir(text_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(text_dir, filename)
                    image_name = filename.split('.')[0]

                    with open(file_path, 'r') as f_in:
                        for line in f_in:
                            stripped_line = line.strip()

                            if stripped_line:
                                result_line = f"{image_name}.jpg | {stripped_line}\n"
                                f_out.write(result_line)
                                break



print(counter)
print("Output file generated successfully!")
