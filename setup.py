import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='label_lines',
    version='0.0.1',
    author='Joshua Perozek',
    author_email='jperozek@gmail.com',
    description='Automatically labels lines with rotation in Matplotlib',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jperozek/label_lines',
    project_urls = {
        "Bug Tracker": "https://github.com/jperozek/label_lines/issues"
    },
    license='MIT',
    packages=['label_lines'],
    install_requires=['math', 'numpy', 'matplotlib', 'datetime', 'scipy', ],
)