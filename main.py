import lyric_processing
import model
import argparse


def run(args):
	X, Y, wordNum, wordToID, words = lyric_processing.pretreatment(args.filename)
	if args.mode == "train":
        print("training...")
        model.train(X, Y, wordNum)
    else:
        print("genrating...")
        lyrics = model.test(wordNum, wordToID, words)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, 
        required=True, help='input the path of the csv file of the lyric.')
    parser.add_argument("-m", "--mode", help = "select mode by 'train' or test",
        choices = ["train", "test"], default = "test")
    
    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()