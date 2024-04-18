from setuptools import setup, find_packages

with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()

# modify entry_points to use command line 
# {COMMAND NAME}={module path}:{function in the module}
setup(
    name="ihbert",
    version="0.0.1",
    description="a repository for in-house BERT implementation",
    author="tadahaya",
    packages=find_packages(),
    install_requires=install_requirements,
    entry_points={
        "console_scripts": [
            "ihbert=ihbert.main:main",
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.9',
    ]
)