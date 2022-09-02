import os


def clean_html_files():
	meta_filename = "docs/html_files_list.txt"
	if not os.path.exists(meta_filename):
		return
	with open(meta_filename, "r") as f:
		files = f.readlines()
	for file in files:
		os.remove(os.path.join(os.getcwd(), "..", "docs", file.strip()))


if __name__ == '__main__':
	clean_html_files()
