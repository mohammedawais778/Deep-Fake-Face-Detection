from setuptools import setup, find_packages

# Read requirements from file
with open('requirements_minimal.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="deepfake-detection",
    version="0.1.0",
    packages=find_packages(include=['src', 'src.*']),
    package_dir={'': '.'},
    install_requires=requirements,
    dependency_links=[
        'https://download.pytorch.org/whl/cpu',
    ],
    package_data={
        'src': [
            'models/*.pth',
            'models/*.h5',
            'models/model/*',
            'utils/*.py',
            'templates/*',
            'static/*',
            'uploads/*'
        ],
    },
    include_package_data=True,
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep Fake Face Detection System",
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'deepfake-detection=src.__main__:main',
        ],
    },
)
