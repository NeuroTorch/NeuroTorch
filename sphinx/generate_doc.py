import os, sys


def generate_doc(path_to_root_dir: str = '.'):
	commands = [
		rf"sphinx-apidoc -f -o {path_to_root_dir}/sphinx/source {path_to_root_dir}/src/neurotorch",
		rf"{path_to_root_dir}\sphinx\make clean html",
		rf"{path_to_root_dir}\sphinx\make html",
	]
	for command in commands:
		print(f"Executing: {command}")
		os.system(command)


if __name__ == '__main__':
	root_dir = sys.argv[1] if len(sys.argv) > 1 else '..'
	generate_doc(root_dir)
