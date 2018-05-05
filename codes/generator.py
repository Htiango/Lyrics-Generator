import argparse
import pickle
from model import test as generate_model
from classification_model import test as classify_model
import tensorflow as tf

pop_model = "./checkpoints/pop"
pop_save = "./generate-param/param-pop-10-test.dat"

rock_model = "./checkpoints/rock"
rock_save = "./generate-param/param-rock-10-test.dat"

rap_model = "./checkpoints/rap"
rap_save = "./generate-param/param-rap-10-test.dat"

classify_model_path = "./checkpoints/textcnn"
classify_save = "./generate-param/param-classify-test.dat"


def run(args):
    genre = args.genre

    if genre == 'pop':
        model_path = pop_model
        data_path = pop_save
    elif genre == 'rock':
        model_path = rock_model
        data_path = rock_save
    elif genre == 'rap':
        model_path = rap_model
        data_path = rap_save
    else:
        print("Unexpected input!")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    print('generating...')

    lyrics = generate_model(data['wordNum'], 
        data['wordToID'], 
        data['words'], 
        model_path=model_path)

    print('\n\n')
    tf.reset_default_graph()

    predicted = classify_model(lyrics[0], classify_save, genre, model_path=classify_model_path)
    print("\n\nOur classification model predict it to be: ")
    print(predicted)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--genre", help = "select the genre to generate a lyric",
        choices = ["pop", "rock", "rap"], required=True)
    
    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()