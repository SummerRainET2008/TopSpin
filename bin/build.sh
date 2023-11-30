# step 1) Change version setup.cfg and src/pyal/__init__.py
# step 2) add __init__ in each folder
 
rm -r dist 
python3 -m build
