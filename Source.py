# # Uncomment to generate sets
# from Sets.SetGenerator import SetGenerator
# generator = SetGenerator()
# generator.generate_sets(2)

# from Sets.SetImporter import SetImporter
# from Sets.SetGenerator import SetGenerator
from Model.Model import Model

all_cards_no = 2
# train_set = SetImporter("{}/poker-hand-{}-{}.data".format(SetGenerator.dir_path, "training", all_cards_no),
# all_cards_no)
# test_set = SetImporter("{}/poker-hand-{}-{}.data".format(SetGenerator.dir_path, "test", all_cards_no), all_cards_no)

# model = Model(train_set)
model = Model()
# neurons_in_hidden_layer = 28
# model.create(neurons_in_hidden_layer)
model.load("{}/model-{}".format(Model.dir_path, all_cards_no), all_cards_no)
# model.train()
# model.save("{}/model-{}".format(Model.dir_path, all_cards_no))
# model.test(test_set)
print(model.predict(['5s', '5d', 'Td', 'Ts', 'Qs']))
