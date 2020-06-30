from __future__ import print_function
entries = []

with open("totals.txt", "r") as file:
    linenumber = 0
    for line in file:

        line = line.strip()
        linenumber += 1
        #print(str(linenumber) + ": " + line)

        if linenumber % 3 != 0:
            #add to people_amnt list
            entries.append(line)

#print(entries)

print(entries[-1])