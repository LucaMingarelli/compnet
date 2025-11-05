rm -dr build
rm -dr dist
rm -dr compnet.egg-info
python3 setup.py sdist bdist_wheel
twine upload dist/*