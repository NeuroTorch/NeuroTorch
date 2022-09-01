import os


def generate_doc():
	commands = [
		r"sphinx-apidoc -f -o ./source ../src/neurotorch",
		r".\make clean html",
		r".\make html",
	]
	for command in commands:
		print(f"Executing: {command}")
		os.system(command)


if __name__ == '__main__':
	generate_doc()
