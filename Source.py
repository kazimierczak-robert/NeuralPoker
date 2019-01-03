from Model import Model
from SetGenerator import SetGenerator
from SetImporter import SetImporter

training_set = SetGenerator()
training_set.generate_sets(3)  # on board

# train_test = SetImporter("Sets\poker-hand-training-5.data", 5)
# test_test = SetImporter("Sets\poker-hand-test-5.data", 5)
# model = Model(train_test)
# model.create()
# model.train()
# model.test(test_test)