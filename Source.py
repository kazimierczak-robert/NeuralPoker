# # Uncomment to generate sets
# from Sets.SetGenerator import SetGenerator
# generator = SetGenerator()
# generator.generate_sets(2)

from Sets.SetImporter import SetImporter
from Sets.SetGenerator import SetGenerator
from Model.Model import Model

all_cards_no = 5
train_set = SetImporter("{}/poker-hand-{}-{}.data".format(SetGenerator.dir_path, "training", all_cards_no), all_cards_no)
test_set = SetImporter("{}/poker-hand-{}-{}.data".format(SetGenerator.dir_path, "test", all_cards_no), all_cards_no)

model = Model(train_set)
# for neurons_in_hidden_layer in range(8, 32, 8):
print("Result for {} neurons in hidden layer".format(28))
# model.create(28)
model.load("{}/{}{}".format(Model.dir_path, "model-", all_cards_no), all_cards_no)
# model.train()
# model.save("{}/{}{}".format(Model.dir_path, "model-", all_cards_no))
# model.test(test_set)
print(model.predict(['5s', '5d', 'Td', 'Ts', 'Qs']))
