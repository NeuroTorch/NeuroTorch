import os
import shutil


def clean_html_files():
	meta_filename = "docs/html_files_list.txt"
	if not os.path.exists(meta_filename):
		return
	with open(meta_filename, "r") as f:
		files = f.readlines()
	for file in files:
		path = os.path.join(os.getcwd(), "docs", file.strip())
		if os.path.exists(path):
			if os.path.isdir(path):
				shutil.rmtree(path)
			else:
				os.remove(path)
	os.remove(meta_filename)


if __name__ == '__main__':
	clean_html_files()
