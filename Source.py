# # Uncomment to generate sets
# from Sets.SetGenerator import SetGenerator
# generator = SetGenerator()
# training_set.generate_sets(5)

from Sets.SetImporter import SetImporter
from Sets.SetGenerator import SetGenerator
from Model import Model

all_cards_no = 5
train_set = SetImporter("{}/{}.data".format(SetGenerator.dir_path, "training", all_cards_no), all_cards_no)
test_set = SetImporter("{}/{}.data".format(SetGenerator.dir_path, "test", all_cards_no), all_cards_no)

model = Model(train_set)
# for neurons_in_hidden_layer in range(8, 32, 8):
print("Result for {} neurons in hidden layer".format(28))
model.create(28)
model.train()
model.test(test_set)
