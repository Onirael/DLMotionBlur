import os

#------------------User input-------------------#

workDirectory = "D:/BachelorCaptures/Capture1"
askConfirm = True
replace = "Test2"
replaceWith = "Capture1"

os.chdir(workDirectory)

exFile = os.listdir()[0]
exRename = exFile.replace(replace, replaceWith)
inConfirm = input(exFile + " will be renamed to : " + exRename + "\nProceed (Y/N) ?")

if (inConfirm in ['y', 'Y']) :
    for file in os.listdir() :
        newName = file.replace(replace, replaceWith)
        os.rename(file, newName)