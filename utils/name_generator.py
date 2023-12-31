import random
from trainer_lib.permutation_grid import Grid

animals = ["Rabbit",
           "Panda",
           "Penguin",
           "Koala",
           "Sloth",
           "Otter",
           "Pug",
           "Kangaroo",
           "Squirrel",
           "Bunny",
           "Puppy",
           "Kitten",
           "Fox",
           "Hamster",
           "Giraffe",
           "Elephant",
           "Turtle",
           "Duck",
           "Chicken",
           "Pig",
           "Goat",
           "Cow",
           "Sheep",
           "Horse",
           "Dog",
           "Cat",
           "Monkey",
           "Bear",
           "Frog",
           "Deer",
           "Raccoon",
           "Llama",
           "Alpaca",
           "Lion",
           "Tiger",
           "Leopard",
           "Cheetah"]
adjectives = ["adorable",
              "adventurous",
              "aggressive",
              "agreeable",
              "alert",
              "alive",
              "amused",
              "angry",
              "annoyed",
              "annoying",
              "anxious",
              "arrogant",
              "ashamed",
              "attractive",
              "average",
              "awful",
              "bad",
              "beautiful",
              "better",
              "bewildered",
              "black",
              "bloody",
              "blue",
              "blue-eyed",
              "blushing",
              "bored",
              "brainy",
              "brave",
              "breakable",
              "bright",
              "busy",
              "calm",
              "careful",
              "cautious",
              "charming",
              "cheerful",
              "clean",
              "clear",
              "clever",
              "cloudy",
              "clumsy",
              "colorful",
              "combative",
              "comfortable",
              "concerned",
              "condemned",
              "confused",
              "cooperative",
              "courageous",
              "crazy",
              "creepy",
              "crowded",
              "cruel",
              "curious",
              "cute",
              "dangerous",
              "dark",
              "dead",
              "defeated",
              "defiant",
              "delightful",
              "depressed",
              "determined",
              "different",
              "difficult",
              "disgusted",
              "distinct",
              "disturbed",
              "dizzy",
              "doubtful",
              "drab",
              "dull",
              ]


def generate_name(length: int, rnd_seed: int | None = None):
    if rnd_seed:
        random.seed(rnd_seed)

    grid = Grid({
        'animal': animals,
        'adjective': adjectives
    })
    indexes = random.sample([i for i in range(len(grid))], length)
    return list(map(lambda x: f"{x['adjective'].capitalize()}_{x['animal'].capitalize()}", grid[indexes]))

