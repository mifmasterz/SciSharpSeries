
using Keras;
using Keras.Layers;
using Keras.Models;
using Numpy;
using KerasNet.Regression;
using Keras.Optimizers;

//tarik data csv
var datasetPath = @"D:\experiment\SciSharpSeries\Dataset\auto-mpg.csv";
var data = KerasNet.Regression.DatasetHelper.LoadAsDataTable(datasetPath);
//hapus kolom
data.Drop(new[] { "car name" });
//one hot encoding
data.OneHotEncoding("origin");
//split training n test data
var (dt_train, dt_test) = data.Split();
//lihat data contoh
data.Head();
//buang kolom y
NDarray y_train = dt_train.Pop("mpg");
//normalisasi dengan z-score
data.Normalization();
//features
NDarray x_train = dt_train.ToNDArray();

//Build sequential model
var model = new Sequential();
model.Add(new Dense(64, activation: "relu", input_shape: new Shape(dt_train.Columns.Count)));
model.Add(new Dense(64, activation: "relu"));
model.Add(new Dense(1));

//Compile and train
model.Compile(optimizer: new Adam(0.001f), loss: "mean_absolute_error");
model.Fit(x_train, y_train, batch_size: 1, epochs: 100, verbose: 1,validation_split:0.2f);

//test
NDarray y_test = dt_test.Pop("mpg");
dt_test.Normalization();
NDarray x_test = dt_test.ToNDArray();

//var score = model.Evaluate(x_test, y_test);
//Console.WriteLine("Test loss:", score[0]);
//Console.WriteLine("Test accuracy:", score[1]);

//test
var res = model.Predict(x_test);
Console.WriteLine("hasil:"+res.ToString());

//Save model and weights
string json = model.ToJson();
File.WriteAllText("model.json", json);
model.SaveWeight("model.h5");

//Load model and weight
var loaded_model = Sequential.ModelFromJson(File.ReadAllText("model.json"));
loaded_model.LoadWeight("model.h5");