import lyric_processing
import model
import argparse

def train(args):
    X, Y, wordNum, wordToID, words = lyric_processing.pretreatment(args.filename)
    print("training...")
    model.train(X, Y, wordNum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, 
        required=True, help='input the path of the csv file of the lyric.')
    
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()