import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seqspawner",
    version="0.0.1",
    author="Tyler Benson",
    author_email="tb19@princeton.edu",
    description="An object-oriented Python library for simulating biological sequences.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tybens/SeqSpawner",
    packages=['seqspawner'],
    insall_requires=['numpy>=1.9', 'unittest2', 'operator', 'abc'],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
