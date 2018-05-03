import argparse
import pickle
import model

pop_model = "./checkpoints/pop"
pop_save = "./generate-param/param-pop-10-test.dat"

rock_model = "./checkpoints/rock"
rock_save = "./generate-param/param-rock-10-test.dat"

rap_model = "./checkpoints/rap"
rap_save = "./generate-param/param-rap-10-test.dat"

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

    lyrics = model.test(data['wordNum'], 
        data['wordToID'], 
        data['words'], 
        model_path=model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--genre", help = "select the genre to generate a lyric",
        choices = ["pop", "rock", "rap"], required=True)
    
    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()