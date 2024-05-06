from distutils.core import setup


setup(
    name='sidm_halos',
    version='0.1',
    description="Jack's tools for computing SIDM dark matter profiles",
    author="Jackson O'Donnell",
    author_email='jacksonhodonnell@gmail.com',
    packages=['sidm_halos', 'sidm_halos.cse', 'sidm_halos.jeans_solver'],
    install_requires=[
        'astropy',
        'numpy', 'scipy', 'matplotlib',
        'colossus',
    ]
)
