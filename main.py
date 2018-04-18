import lyric_processing
import model
import argparse

import lyric_process_old
import pickle

param_saving_path = "./data/param.dat"

def run(args):
    
    if args.mode == "train":
        # X, Y, wordNum, wordToID, words = lyric_processing.pretreatment(args.filename)
        # data = {'X': X, "Y":Y, "wordNum":wordNum, "wordToID": wordToID, "words":words}

        # with open(param_saving_path, 'wb') as f:
        #     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        with open(param_saving_path, 'rb') as f:
            data = pickle.load(f)

        print("training...")
        model.train(data['X'], data['Y'], data['wordNum'])
    else:
        with open(param_saving_path, 'rb') as f:
            data = pickle.load(f)
        print("genrating...")
        lyrics = model.test(data['wordNum'], data['wordToID'], data['words'])



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