import os


def make_html_file_list():
	meta_filename = "./docs/html_files_list.txt"
	os.makedirs(os.path.dirname(meta_filename), exist_ok=True)
	files = os.listdir(os.path.join('sphinx', "build", "html"))
	with open(meta_filename, "w") as f:
		f.write("\n".join(files))
	

if __name__ == '__main__':
	make_html_file_list()
