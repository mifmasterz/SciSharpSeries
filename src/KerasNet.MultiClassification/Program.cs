
using Keras;
using Keras.Layers;
using Keras.Models;
using Numpy;
using Keras.Optimizers;
using ML.Tools;
using Keras.Utils;
using SliceAndDice;

var datasetPath = $"{FileHelpers.AppDirectory}\\..\\..\\..\\..\\Dataset\\iris.csv";
Console.WriteLine(datasetPath);

//tarik data csv
var data = DatasetHelper.LoadAsDataTable(datasetPath);
var labels = data.ToCategory("iris_type");
//split training n test data
var (dt_train, dt_test) = data.Split();
//lihat data contoh
data.Head();
//buang kolom y
NDarray y_train = dt_train.Pop("iris_type");
//normalisasi dengan z-score
data.Normalization();
//features
NDarray x_train = dt_train.ToNDArray();

//Build sequential model
var model = new Sequential();
model.Add(new Dense(128, activation: "relu", input_shape: new Keras.Shape(dt_train.Columns.Count)));
model.Add(new Dense(10, activation: "relu"));
model.Add(new Dense(3,activation:"softmax")); //disesuaikan dengan jumlah class labelnya

//Compile and train
model.Compile(optimizer: new Adam(), loss: "sparse_categorical_crossentropy", metrics: new string[] { "accuracy" } );
model.Fit(x_train, y_train, batch_size: 1, epochs: 100, verbose: 1, validation_split: 0.2f);

//test
var jawaban = dt_test.ToStringArray("iris_type");
NDarray y_test = dt_test.Pop("iris_type");
dt_test.Normalization();
NDarray x_test = dt_test.ToNDArray();

var score = model.Evaluate(x_test, y_test);
Console.WriteLine("Test loss: {0:n2}", score[0]);
Console.WriteLine("Test accuracy: {0:n2}", score[1]);

//test
var res = model.Predict(x_test);
//parse ndarray to float
ArraySlice<float> ts = new ArraySlice<float>(res.GetData<float>());
//slice and dice
var xx = ts.Reshape(3, dt_test.Rows.Count).Chunk(3);
var counter = 0;
foreach(var x in xx)
{
    var max = x.Max();
    var index = Array.IndexOf(x, max);
    Console.WriteLine($"test no.{counter+1} = {labels[index]} / {(index ==  int.Parse(jawaban[counter]) ? "benar":"salah")}");
    counter++;
}

//Save model and weights
string json = model.ToJson();
File.WriteAllText("model.json", json);
model.SaveWeight("model.h5");

//Load model and weight
var loaded_model = Sequential.ModelFromJson(File.ReadAllText("model.json"));
loaded_model.LoadWeight("model.h5");