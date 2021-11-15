from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='abstractions-pkg-aimedic',
    version='0.1',
    description="Abstractions for AIMedic's training components.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.aimedic.co/soroush.moazed/abstractions.git",
    classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # packages=['abstractions'],
    python_requires=">=3.6"
)