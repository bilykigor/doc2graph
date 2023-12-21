import os

from setuptools import find_packages, setup
#from setuptools_git_versioning import version_from_git
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))

def parse_requirements(file_content):
    lines = file_content.splitlines()
    return [line.strip() for line in lines if line and not line.startswith("#")]

with open(os.path.join(HERE, "requirements.txt")) as f:
    requirements = parse_requirements(f.read())

# setup(
#     packages=['doc2graph'],
#     package_dir={'doc2graph': 'src'},
#     setuptools_git_versioning={
#         "enabled": True,
#     },
#     setup_requires=["setuptools_git_versioning"],
# )

def read(fname):
    """
    Args:
        fname:
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Declare minimal set for installation
required_packages = [
    #"boto3>=1.20.21,<2.0",
    "pandas",
]

setup(
    name="doc2graph",
    version="3.3.5",#version_from_git(),
    description="Repo to transform Documents to Graphs, performing several tasks on them.",
    long_description_content_type="text/markdown",
    long_description=read('README.md'),
    packages=find_packages(),
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob("doc2graph/*.py")],
    include_package_data=True,
    author="Ihor Bilyk fork Andrea Gemelli",
    url="https://github.com/bilykigor/doc2graph",
    license="Apache License 2.0",
    keywords="document analysis graph",
    python_requires=">= 3.7",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=requirements#required_packages
)
