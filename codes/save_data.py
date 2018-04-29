import lyric_processing
import model
import argparse

# import lyric_process_old
import pickle

def save(args):

    input_path= args.input
    param_saving_path = args.output
    batch_size = args.batch_size

    X, Y, wordNum, wordToID, words = lyric_processing.pretreatment(input_path,batch_size)
    data = {'X': X, "Y":Y, "wordNum":wordNum, 
        "wordToID": wordToID, "words":words, 'batch_size':batch_size}

    with open(param_saving_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    print("model saved!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, 
        required=True, help='input the path of the csv file of the lyric.')
    parser.add_argument('-o', '--output', type=str, 
        required=True, help='output path of param.')
    parser.add_argument('-n', '--batch_size', type=int, 
    	required=True, help='batch size')

    args = parser.parse_args()
    save(args)


if __name__ == "__main__":
    main()