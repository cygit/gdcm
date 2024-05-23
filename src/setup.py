from setuptools import setup, find_packages

setup(
    name='gdcm',
    version='0.1',
    author_email='maochfe@gmail.com',
    packages=find_packages(where='src'),
    py_modules=['main'],
    install_requires=['click', 'nltk', 'scikit-learn', 'pandas', 'torch', 'scipy', 'pyLDAvis', 'datasets', 'numpy', 'django', 'gensim', 'wordcloud', 'matplotlib'],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'gdcm = cli:gdcm'
        ]
    }
)
