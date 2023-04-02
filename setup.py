from setuptools import find_packages, setup
from typing import List

HYPHEN_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    """
     The function returns the list of requirements
    """
    requirements = []
    with open(file_path) as obj:
        requirements_list = obj.readlines()
        requirements_list = [ req.replace("\n", "") for req in requirements_list]
        requirements_list.remove(HYPHEN_DOT)

    return requirements_list

setup(
    name             = 'ml_project',
    version          = '0.0.1',
    author           = 'demo',
    author_email     = 'demo@gmail.com',
    packages         = find_packages(),
    install_requires = get_requirements('requirements.txt')
    # install_requires = ['pandas', 'numpy', 'seaborn']
)



