
import sys

ogfilename = str(sys.argv[1])



def csvname(ogfilename):

    ogmain = ogfilename.split('.')[1]
    csvfile = str(ogmain) + '.csv'


    return csvfile




if __name__ == "__main__":

    csvfile = csvname(ogfilename=ogfilename)

    print csvfile