from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='abstractions-aimedic',
    version='0.1.5',
    license='MIT',
    author='Soroush Moazed',
    author_email='soroush.moazed@gmail.com',
    description="Abstractions for AIMedic's training components.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.aimedic.co/soroush.moazed/abstractions.git",
    download_url="https://github.com/iamsoroush/abstractions/archive/refs/tags/v_0.1.4.tar.gz",
    classifiers=[
            'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
            'Intended Audience :: Developers',      # Define that your audience are developers
            'Topic :: Software Development :: Build Tools',
            'License :: OSI Approved :: MIT License',   # Again, pick a license
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'pyyaml',
        'tensorflow>=2.5',
        'tqdm',
        'pandas',
        'mlflow'
    ],
    # packages=['abstractions'],
    python_requires=">=3.6"
)