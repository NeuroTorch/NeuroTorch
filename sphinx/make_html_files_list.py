import os


def make_html_file_list():
	meta_filename = "docs/html_files_list.txt"
	files = os.listdir(os.path.join(os.getcwd(), "build", "html"))
	with open(meta_filename, "w+") as f:
		f.write("\n".join(files))
	

if __name__ == '__main__':
	make_html_file_list()
