# NeuralPoker

Project is about neural network determining hand and probability of winning with hand and game stage (*preflop*, *flop*, *turn*, *river*) in Poker Texas Hold'em. Requires python3.

## External Libraries

* **Deuces** - used to determine hand and their strength. Available on: https://github.com/worldveil/deuces and compatible with python2 
* **Monte-Carlo** - used to determine probability of winning with hand and game stage using Monte Carlo method. Available on: https://github.com/Blind0ne/Monte-Carlo. Requires **deuces** library.

## Generating training and test set

To generate **training** and **test set** use the *SetGenerator.py* file. <code>generate_sets</code> method takes parameters: the number of all card at given stage: <code>all_cards_no(2, 5, 6, 7)</code>. The number of sets depends on the parameter:

  <table>
  <tr>
    <th>Parameter</th>
    <th>Test set</th>
    <th>Training set</th>
  </tr>
  <tr>
    <td>2</td>
    <td>133<br></td>
    <td>1195</td>
  </tr>
  <tr>
    <td>5</td>
    <td>3979</td>
    <td>35803</td>
  </tr>
  <tr>
    <td>6</td>
    <td>3979</td>
    <td>35803</td>
  </tr>
  <tr>
    <td>7</td>
    <td>3979</td>
    <td>35803</td>
  </tr>
</table>

Method prints to STDOUT number of the generated sample (trening, test)

Part of the sample training set for the parameter <code>all_cards_no = 2</code>:
<pre>
<code>
2,4,2,6,0.20840000000000003
1,6,4,13,0.23440000000000005  
2,6,4,13,0.25
3,12,4,12,0.5028
1,6,3,6,0.24680000000000002
</code>
</pre>
Card is represented by pair: suit and rank. Two pairs are cards in hand. Last number is the probability of winnig.

Part of the sample training set for the parameter <code>all_cards_no = 6</code>:
<pre>
<code>
1,5,4,5,1,12,3,7,2,13,4,1,8,0.12680000000000002
3,4,4,4,4,5,4,9,2,4,1,4,2,0.9996
1,6,4,6,1,8,2,8,2,1,2,2,7,0.28080000000000005
1,2,2,5,4,7,3,6,3,13,1,9,9,0.03520000000000001
4,4,3,7,4,6,3,9,4,10,1,7,8,0.11599999999999999
</code>
</pre>
Card is represented by pair: suit and rank. Last two numbers are: the strongest hand and probability of winning. The first two pairs are the cards in hand, the others - on board.
<pre>
<code>
from Sets.SetGenerator import SetGenerator
all_cards_no = 5
generator = SetGenerator()
generator.generate_sets(all_cards_no)
</code>
</pre>

## Loading of the training and test set
You must import the <code>SetImporter.py</code> and <code>SetGenerator.py</code> files. The constructor of the <code>SetImporter</code> class is two-argument: the path to the file with the set and the number of all cards at a given stage of the game.
<pre>
<code>
from Sets.SetImporter import SetImporter
from Sets.SetGenerator import SetGenerator
all_cards_no = 5
train_set = SetImporter("{}/{}.data".format(SetGenerator.dir_path, "training", all_cards_no), all_cards_no)
test_set = SetImporter("{}/{}.data".format(SetGenerator.dir_path, "test", all_cards_no), all_cards_no)
</code>
</pre>

## Model of the neural network
To train the model and test it, execute the code below. The <code>create</code> method parameter is the number of hidden layer neurons.
<pre>
<code>
from Sets.SetImporter import SetImporter
from Sets.SetGenerator import SetGenerator
from Model.Model import Model

all_cards_no = 5
train_set = SetImporter("{}/poker-hand-{}-{}.data".format(SetGenerator.dir_path, "training", all_cards_no), all_cards_no)
test_set = SetImporter("{}/poker-hand-{}-{}.data".format(SetGenerator.dir_path, "test", all_cards_no), all_cards_no)

model = Model(train_set)
model.create(28)
model.train()
model.test(test_set)
</code>
</pre>
The <code>create</code> method creates a neural network model using the **tensorflow** library. The network consists of an input layer with 17 neurons (13 rank + 4 suits), one hidden Dense layer, which uses the relu activation function. Depending on the selected stage of the game (2 or 5, 6, 7), the network has one (for 2) or 2 output layers (for 5, 6, 7). The output layer with one neuron determining the probability of winning (regression) uses the sigmoid activation function. The output layer with 10 neurons defining the hand (classification) uses the softmax activation function. The following metrics were used: acc (classification), mse (regression).
<pre>
<code>
def compile(self):
    if self.all_cards == 2:
        # metrics for regression
        self.model.compile(optimizer=tf.train.AdamOptimizer(), loss='mse', metrics=['mse'])
    else:
        # metrics for classification and regression
        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss={'output_layer_1': 'sparse_categorical_crossentropy', 'output_layer_2': 'mse'},
                           metrics={'output_layer_1': 'acc', 'output_layer_2': 'mse'})

def create(self, neurons_in_hidden_layer):
    if self.dataset is not None:
        self.model = keras.models.Sequential()
        # 17 = 13 ranks + 4 suits
        input_layer = keras.layers.Input((17&ast;self.dataset.all_cards,), name="input_layer")
        hidden_layer = keras.layers.Dense(neurons_in_hidden_layer, activation=tf.nn.relu, name="hidden_layer")(input_layer)
        # 1 output in NN
        if self.dataset.all_cards == 2:
            output_layer = keras.layers.Dense(1, activation=tf.nn.sigmoid, name="output_layer")(hidden_layer)
            self.model = keras.models.Model(input_layer, output_layer)
        # 2 outputs in NN
        else:
            output_1 = keras.layers.Dense(10, activation=tf.nn.softmax, name="output_layer_1")(hidden_layer)
            output_2 = keras.layers.Dense(1, activation=tf.nn.sigmoid, name="output_layer_2")(hidden_layer)
            self.model = keras.models.Model(input_layer, [output_1, output_2])
        self.compile()
</code>
</pre>

## Configuration testing
* <code>all_cards_no = 2</code>

<table>
  <tr>
    <th>neurons_no</th>
    <th>loss</th>
    <th>mean_squared_error</th>
  </tr>
  <tr>
    <td>8</td>
    <td>0.0002728728868532926<br></td>
    <td>0.0002728728868532926</td>
  </tr>
  <tr>
    <td>16</td>
    <td>0.00019329327915329486</td>
    <td>0.00019329327915329486</td>
  </tr>
  <tr>
    <td>24</td>
    <td>0.0002716764865908772</td>
    <td>0.0002716764865908772</td>
  </tr>
  <tr>
    <td>28</td>
    <td>0.00016631107428111136</td>
    <td>0.00016631107428111136</td>
  </tr>
  <tr>
    <td>29</td>
    <td>0.00021212051797192544</td>
    <td>0.00021212051797192544</td>
  </tr>
  <tr>
    <td>30</td>
    <td>0.00021880857821088284</td>
    <td>0.00021880857821088284</td>
  </tr>
  <tr>
    <td>32</td>
    <td>0.00019229181634727865</td>
    <td>0.00019229181634727865</td>
  </tr>
</table>

* <code>all_cards_no = 5</code>

<table>
  <tr>
    <th>neurons_no</th>
    <th>loss</th>
    <th>output_layer_1_loss</th>
    <th>output_layer_2_loss</th>
    <th>output_layer_1_acc</th>
    <th>output_layer_2_mean_squared_error</th>
  </tr>
  <tr>
    <td>8</td>
    <td>0.8569035530090332</td>
    <td>0.8307963013648987</td>
    <td>0.026107240468263626</td>
    <td>0.6337355375289917</td>
    <td>0.026107240468263626</td>
  </tr>
  <tr>
    <td>16</td>
    <td>0.18359050154685974</td>
    <td>0.1657908409833908</td>
    <td>0.017799654975533485</td>
    <td>0.9472096562385559</td>
    <td>0.017799654975533485</td>
  </tr>
  <tr>
    <td>24</td>
    <td>0.23562563955783844</td>
    <td>0.2202180176973343</td>
    <td>0.015407627448439598</td>
    <td>0.9678230285644531</td>
    <td>0.015407627448439598</td>
  </tr>
  <tr>
    <td>28</td>
    <td>0.24058467149734497</td>
    <td>0.2283773422241211</td>
    <td>0.012207326479256153</td>
    <td>0.9690799117088318</td>
    <td>0.012207326479256153</td>
  </tr>
  <tr>
    <td>29</td>
    <td>0.29240182042121887</td>
    <td>0.2786564528942108</td>
    <td>0.01374536007642746</td>
    <td>0.9658119678497314</td>
    <td>0.01374536007642746</td>
  </tr>
  <tr>
    <td>30</td>
    <td>0.35394394397735596</td>
    <td>0.34037789702415466</td>
    <td>0.013566038571298122</td>
    <td>0.9615384340286255</td>
    <td>0.013566038571298122</td>
  </tr>
  <tr>
    <td>32</td>
    <td>0.39161327481269836</td>
    <td>0.37983229756355286</td>
    <td>0.011780978180468082</td>
    <td>0.9610356688499451</td>
    <td>0.011780978180468082</td>
  </tr>
</table>

* <code>all_cards_no = 6</code>

<table>
  <tr>
    <th>neurons_no</th>
    <th>loss</th>
    <th>output_layer_1_loss</th>
    <th>output_layer_2_loss</th>
    <th>output_layer_1_acc</th>
    <th>output_layer_2_mean_squared_error</th>
  </tr>
  <tr>
    <td>8</td>
    <td>1.1386266946792603</td>
    <td>1.100320816040039</td>
    <td>0.03830586373806</td>
    <td>0.5389643311500549</td>
    <td>0.03830586373806</td>
  </tr>
  <tr>
    <td>16</td>
    <td>0.5025573968887329</td>
    <td>0.4704950153827667</td>
    <td>0.0320623554289341</td>
    <td>0.8627451062202454</td>
    <td>0.0320623554289341</td>
  </tr>
  <tr>
    <td>24</td>
    <td>0.2888332009315491</td>
    <td>0.2606068551540375</td>
    <td>0.02822634018957615</td>
    <td>0.9263448715209961</td>
    <td>0.02822634018957615</td>
  </tr>
  <tr>
    <td>28</td>
    <td>0.33630427718162537</td>
    <td>0.3104614317417145</td>
    <td>0.025842837989330292</td>
    <td>0.9281045794487</td>
    <td>0.025842837989330292</td>
  </tr>
  <tr>
    <td>29</td>
    <td>0.3946927785873413</td>
    <td>0.3673554062843323</td>
    <td>0.02733737602829933</td>
    <td>0.9263448715209961</td>
    <td>0.02733737602829933</td>
  </tr>
  <tr>
    <td>30</td>
    <td>0.5774040818214417</td>
    <td>0.5508595705032349</td>
    <td>0.026544496417045593</td>
    <td>0.9301156401634216</td>
    <td>0.026544496417045593</td>
  </tr>
  <tr>
    <td>32</td>
    <td>0.44998517632484436</td>
    <td>0.4231707453727722</td>
    <td>0.02681444026529789</td>
    <td>0.9147812724113464</td>
    <td>0.02681444026529789</td>
  </tr>
</table>

* <code>all_cards_no = 7</code>

<table>
  <tr>
    <th>neurons_no</th>
    <th>loss</th>
    <th>output_layer_1_loss</th>
    <th>output_layer_2_loss</th>
    <th>output_layer_1_acc</th>
    <th>output_layer_2_mean_squared_error</th>
  </tr>
  <tr>
    <td>8</td>
    <td>1.3301163911819458</td>
    <td>1.2595701217651367</td>
    <td>0.0705462396144867</td>
    <td>0.47561588883399963</td>
    <td>0.0705462396144867</td>
  </tr>
  <tr>
    <td>16</td>
    <td>0.9917051196098328</td>
    <td>0.928206741809845</td>
    <td>0.063498355448246</td>
    <td>0.6312217116355896</td>
    <td>0.063498355448246</td>
  </tr>
  <tr>
    <td>24</td>
    <td>0.44883644580841064</td>
    <td>0.39074623584747314</td>
    <td>0.058090221136808395</td>
    <td>0.8637506365776062</td>
    <td>0.058090221136808395</td>
  </tr>
  <tr>
    <td>28</td>
    <td>0.6418846845626831</td>
    <td>0.580848217010498</td>
    <td>0.06103646755218506</td>
    <td>0.8619909286499023</td>
    <td>0.06103646755218506</td>
  </tr>
  <tr>
    <td>29</td>
    <td>0.6029195189476013</td>
    <td>0.5420128703117371</td>
    <td>0.06090662255883217</td>
    <td>0.8501759767532349</td>
    <td>0.06090662255883217</td>
  </tr>
  <tr>
    <td>30</td>
    <td>0.5111713409423828</td>
    <td>0.4530506730079651</td>
    <td>0.05812065303325653</td>
    <td>0.8866264224052429</td>
    <td>0.05812065303325653</td>
  </tr>
  <tr>
    <td>32</td>
    <td>0.571366548538208</td>
    <td>0.5106329917907715</td>
    <td>0.060733549296855927</td>
    <td>0.8692810535430908</td>
    <td>0.060733549296855927</td>
  </tr>
</table>

## Using a neural network
### Save the model to files
<code>model.save("{}/model-{}".format(Model.dir_path, all_cards_no))</code>

### Reading the model from files
<code>model.load("{}/model-{}".format(Model.dir_path, all_cards_no), all_cards_no)</code>

### Prediction of results
Hands are marked with a number. Cards are represented by concatenation of characters: suit and rank.

<div>
<div>
<table>
  <tr>
    <th>Hand number<br></th>
    <th>Hand<br></th>
  </tr>
  <tr>
    <td>0</td>
    <td>Royal Straight Flush</td>
  </tr>
  <tr>
    <td>1<br></td>
    <td>Straight Flush</td>
  </tr>
  <tr>
    <td>2</td>
    <td>Four of a Kind</td>
  </tr>
  <tr>
    <td>3</td>
    <td>Full House</td>
  </tr>
  <tr>
    <td>4</td>
    <td>Flush</td>
  </tr>
  <tr>
    <td>5</td>
    <td>Straight</td>
  </tr>
  <tr>
    <td>6</td>
    <td>Three of a Kind</td>
  </tr>
  <tr>
    <td>7</td>
    <td>Two Pair</td>
  </tr>
  <tr>
    <td>8</td>
    <td>Pair</td>
  </tr>
  <tr>
    <td>9</td>
    <td>High Card</td>
  </tr>
</table>    
</div>
<div>
<table>
  <tr>
    <th>Rank</th>
    <th>Char</th>
  </tr>
  <tr>
    <td>2,3,...,9</td>
    <td>2,3,...,9</td>
  </tr>
  <tr>
    <td>10<br></td>
    <td>10</td>
  </tr>
  <tr>
    <td>Jack</td>
    <td>J</td>
  </tr>
  <tr>
    <td>Queen</td>
    <td>Q</td>
  </tr>
  <tr>
    <td>King</td>
    <td>K</td>
  </tr>
  <tr>
    <td>Ace</td>
    <td>A</td>
  </tr>
</table>
    </div>
<div>
<table>
  <tr>
    <th>Suit</th>
    <th>Char</th>
  </tr>
  <tr>
    <td>Spades</td>
    <td>s</td>
  </tr>
  <tr>
    <td>Hearts</td>
    <td>h</td>
  </tr>
  <tr>
    <td>Diamonds</td>
    <td>d</td>
  </tr>
  <tr>
    <td>Clubs</td>
    <td>c</td>
  </tr>
</table>
    </div>
</div>

The prediction for the game stage with 2 cards in hand returns the probability of winning. For other stages, the hand number (the strongest) is additionally returned.

<code>print(model.predict(['Qs', 'Qh', 'Qd', 'Kc', 'Kh']))</code>

## An example usage scenario

<pre>
<code>
from Model.Model import Model

all_cards_no = 5
model = Model()
model.load("{}/model-{}".format(Model.dir_path, all_cards_no), all_cards_no)
print(model.predict(['Qs', 'Qh', 'Qd', 'Kc', 'Kh']))
</code>
</pre>

(3, 0.89730614)

## Attributions
- https://github.com/worldveil/deuces
- https://github.com/Blind0ne/Monte-Carlo

## Credits
* Monika Grądzka
* Robert Kazimierczak
