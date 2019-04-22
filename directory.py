import os

def check(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)



# Set the directory you want to start from
rootDir = '.'
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    for fname in fileList:
        print('\t%s' % fname)

# https://www.pythoncentral.io/how-to-traverse-a-directory-tree-in-python-guide-to-os-walk/