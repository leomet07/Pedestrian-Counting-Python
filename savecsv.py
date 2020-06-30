import csv

def savetocsv(fname, row, header):
    # add row to CSV file
    append = False
    
    print(row)
    try:
        with open(fname,'r') as userFile:
            userFileReader = csv.reader(userFile)
            #read the first row
            for cyclerow in userFileReader:
                if cyclerow == header:
                    append = True
                break
    except FileNotFoundError:
        print("File not found")
    #append = False
    if append:
        print("Appending")
        with open(fname, "a", newline='') as f:
            writer = csv.writer(f)
            print('writer made')
            writer.writerow(row)
            print('row has been written')
        
    else:
        print("Writing..")
        with open(fname, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)

'''
from datetime import datetime
currentDT = datetime.now()
print (str(currentDT))
#making a csv file with the data collected
mrow = [str(currentDT), str(5),str(89), str("bruh")]
print(mrow)
mheader = ["BRo","Amount_past", "Frame_count", "VidName"]
print("sending csv command")
savetocsv("output.csv", mrow , mheader)
'''