import csv

print("Output\n")

h = ['0'for i in range(6)]
with open("trainingdata.csv") as f:
    data = csv.reader(f)
    data = list(data)

    print("The +ve examples are:")
    for i in data:
        if i[-1] == "Yes":
            print(i)

    print("\nThe steps of Algorithm are:")
    for i in data:
        if i[-1] == "Yes":
            for j in range(6):
                if h[j] == '0':
                    h[j] = i[j]
                elif h[j] != i[j]:
                    h[j] = '?'
                    print(h)

    print("\nFinal specific hypothesis:\n", h)
