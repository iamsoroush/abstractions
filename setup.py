from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='abstractions-aimedic',
    version='0.1.15',
    license='MIT',
    author='Soroush Moazed',
    author_email='soroush.moazed@gmail.com',
    description="Abstractions for AIMedic's training pipeline.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.aimedic.co/soroush.moazed/abstractions.git",
    entry_points='''
            [console_scripts]
            submit=abstractions.submit_training_job:main
            train=abstractions.train:main
        ''',
    download_url="https://github.com/iamsoroush/abstractions/archive/refs/tags/v_0.1.9.tar.gz",
    classifiers=[
            'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
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
        # 'tensorflow>=2.2',
        'tqdm',
        'pandas',
        'mlflow'
    ],
    # packages=['abstractions'],
    python_requires=">=3.6"
)
