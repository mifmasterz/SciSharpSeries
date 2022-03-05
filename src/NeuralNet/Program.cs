using ML.Tools;
using NumSharp;
using System.Diagnostics;
namespace neuralnet
{

    #region cost functions

    public abstract class BaseCost
    {
        public float Epsilon = 1e-7f;

        public string Name { get; set; }

        public BaseCost(string name)
        {
            Name = name;
        }

        public abstract NDArray Forward(NDArray preds, NDArray labels);

        public abstract NDArray Backward(NDArray preds, NDArray labels);
    }

    public class BinaryCrossEntropy : BaseCost
    {
        public BinaryCrossEntropy() : base("binary_crossentropy")
        {

        }

        public override NDArray Forward(NDArray preds, NDArray labels)
        {
            //ToDo: np.clip
            //var output = Clip(preds, Epsilon, 1 - Epsilon);
            var output = preds;
            output = np.mean(-(labels * np.log(output) + (1 - labels) * np.log(1 - output)));
            return output;
        }

        public override NDArray Backward(NDArray preds, NDArray labels)
        {
            //ToDo: np.clip
            //var output = Clip(preds, Epsilon, 1 - Epsilon);
            var output = preds;
            return (output - labels) / (output * (1 - output));
        }
    }

    public class CategoricalCrossentropy : BaseCost
    {
        public CategoricalCrossentropy() : base("categorical_crossentropy")
        {

        }

        public override NDArray Forward(NDArray preds, NDArray labels)
        {
            //ToDo: np.clip
            //var output = Clip(preds, Epsilon, 1 - Epsilon);
            var output = preds;
            output = np.mean(-(labels * np.log(output)));
            return output;
        }

        public override NDArray Backward(NDArray preds, NDArray labels)
        {
            //ToDo: np.clip
            //var output = Clip(preds, Epsilon, 1 - Epsilon);
            var output = preds;
            return (output - labels) / output;
        }
    }
    public class MeanSquaredError : BaseCost
    {
        public MeanSquaredError() : base("mean_squared_error")
        {

        }

        public override NDArray Forward(NDArray preds, NDArray labels)
        {
            var error = preds - labels;
            return np.mean(np.power(error, 2));
        }

        public override NDArray Backward(NDArray preds, NDArray labels)
        {
            float norm = 2 / (float)preds.shape[0];
            return norm * (preds - labels);
        }
    }
    #endregion

    #region activation functions
    public class BaseActivation : BaseLayer
    {
        public BaseActivation(string name) : base(name)
        {

        }

        public static BaseActivation Get(string name)
        {
            BaseActivation baseActivation = null;
            switch (name)
            {
                case "relu":
                    baseActivation = new ReLU();
                    break;
                case "sigmoid":
                    baseActivation = new Sigmoid();
                    break;
                default:
                    break;
            }

            return baseActivation;
        }
    }

    public class ReLU : BaseActivation
    {
        public ReLU() : base("relu")
        {

        }

        public override void Forward(NDArray x)
        {
            base.Forward(x);

            NDArray matches = x > 0;
            Output = matches * x;
        }

        public override void Backward(NDArray grad)
        {
            InputGrad = grad * (NDArray)(Input > 0);
        }
    }
    public class Sigmoid : BaseActivation
    {
        public Sigmoid() : base("sigmoid")
        {

        }

        public override void Forward(NDArray x)
        {
            base.Forward(x);
            //ToDo: np.exp
            //Output = 1 / (1 + Exp(-x));
        }

        public override void Backward(NDArray grad)
        {
            InputGrad = grad * Output * (1 - Output);
        }
    }
    public class Softmax : BaseActivation
    {
        public Softmax() : base("softmax")
        {

        }

        public override void Forward(NDArray x)
        {
            base.Forward(x);
            //ToDo: Implement np.exp
            //Output = 1 / (1 + Exp(-x));
        }

        public override void Backward(NDArray grad)
        {
            InputGrad = grad * Output * (1 - Output);
        }
    }
    #endregion

    #region layers
    /// <summary>
    /// Base class for the layers with predefined variables and functions
    /// </summary>
    public abstract class BaseLayer
    {
        /// <summary>
        /// Name of the layer
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Input for the layer
        /// </summary>
        public NDArray Input { get; set; }

        /// <summary>
        /// Output after forwarding the input across the neurons
        /// </summary>
        public NDArray Output { get; set; }

        /// <summary>
        /// Trainable parameters list, eg, weight, bias
        /// </summary>
        public Dictionary<string, NDArray> Parameters { get; set; }

        /// <summary>
        /// Gradient of the Input
        /// </summary>
        public NDArray InputGrad { get; set; }

        /// <summary>
        /// List of all parameters gradients calculated during back propagation.
        /// </summary>
        public Dictionary<string, NDArray> Grads { get; set; }

        /// <summary>
        /// Base layer instance
        /// </summary>
        /// <param name="name"></param>
        public BaseLayer(string name)
        {
            Name = name + Util.GetNext();
            Parameters = new Dictionary<string, NDArray>();
            Grads = new Dictionary<string, NDArray>();
        }

        /// <summary>
        /// Virtual forward method to perform calculation and move the input to next layer
        /// </summary>
        /// <param name="x"></param>
        public virtual void Forward(NDArray x)
        {
            Input = x;
        }

        /// <summary>
        /// Calculate the gradient of the layer. Usually a prtial derivative implemenation of the forward algorithm
        /// </summary>
        /// <param name="grad"></param>
        public virtual void Backward(NDArray grad)
        {

        }

        public void PrintParams(bool printGrads = true)
        {
            foreach (var item in Parameters)
            {
                Console.WriteLine(item.Value.ToString());
                if (printGrads && Grads.ContainsKey(item.Key))
                {
                    Console.WriteLine(Grads[item.Key].ToString());
                }
            }
        }
    }
    /// <summary>
    /// Fully connected layer
    /// </summary>
    public class FullyConnected : BaseLayer
    {
        /// <summary>
        /// Number of incoming input features
        /// </summary>
        public int InputDim { get; set; }

        /// <summary>
        /// Number of neurons for this layers
        /// </summary>
        public int OutNeurons { get; set; }

        /// <summary>
        /// Non Linear Activation function for this layer of neurons. All neurons will have the same function
        /// </summary>
        public BaseActivation Activation { get; set; }

        /// <summary>
        /// Constructor with in and out parametes
        /// </summary>
        /// <param name="in">Number of incoming input features</param>
        /// <param name="out">Number of neurons for this layers</param>
        public FullyConnected(int input_dim, int output_neurons, string act = "") : base("fc")
        {
            Parameters["w"] = np.random.normal(0.5, 1, input_dim, output_neurons);
            InputDim = input_dim;
            OutNeurons = output_neurons;

            Activation = BaseActivation.Get(act);
        }

        /// <summary>
        /// Forward the input data by performing calculation across all the neurons, store it in the Output to be accessible by next layer.
        /// </summary>
        /// <param name="x"></param>
        public override void Forward(NDArray x)
        {
            base.Forward(x);
            Console.WriteLine(Parameters["w"].shape[0].ToString());
            Console.WriteLine(Parameters["w"].shape[1].ToString());

            Output = np.dot(x, Parameters["w"]);

            if (Activation != null)
            {
                Activation.Forward(Output);
                Output = Activation.Output;
            }
        }

        /// <summary>
        /// Calculate the gradient of the layer. Usually a prtial derivative implemenation of the forward algorithm
        /// </summary>
        /// <param name="grad"></param>
        public override void Backward(NDArray grad)
        {
            if (Activation != null)
            {
                Activation.Backward(grad);
                grad = Activation.InputGrad;
            }

            InputGrad = np.dot(grad, Parameters["w"].transpose());
            Grads["w"] = np.dot(Input.transpose(), grad);
        }
    }
    #endregion

    #region metrics
    public abstract class BaseMetric
    {
        public string Name { get; set; }

        public BaseMetric(string name)
        {
            Name = name;
        }

        public abstract NDArray Calculate(NDArray preds, NDArray labels);
    }
    public class Accuracy : BaseMetric
    {
        public Accuracy() : base("accurary")
        {
        }

        public override NDArray Calculate(NDArray preds, NDArray labels)
        {
            var pred_idx = np.argmax(preds);
            var label_idx = np.argmax(labels);

            return np.mean(pred_idx == label_idx);
        }
    }
    public class BinaryAccuracy : BaseMetric
    {
        public BinaryAccuracy() : base("binary_accurary")
        {
        }

        public override NDArray Calculate(NDArray preds, NDArray labels)
        {
            //ToDo: np.round and np.clip
            //var output = Round(Clip(preds, 0, 1));
            //return np.mean(output == labels);
            return null;
        }
    }
    public class MeanAbsoluteError : BaseMetric
    {
        public MeanAbsoluteError() : base("mean_absolute_error")
        {

        }

        public override NDArray Calculate(NDArray preds, NDArray labels)
        {
            var error = preds - labels;
            return np.mean(np.abs(error));
        }
    }
    #endregion

    #region optimizer
    public abstract class BaseOptimizer
    {
        public float Epsilon = 1e-7f;

        /// <summary>
        /// Gets or sets the name of the optimizer function
        /// </summary>
        /// <value>
        /// The name.
        /// </value>
        public string Name { get; set; }

        /// <summary>
        /// Gets or sets the learning rate for the optimizer.
        /// </summary>
        /// <value>
        /// The learning rate.
        /// </value>
        public float LearningRate { get; set; }

        /// <summary>
        /// Parameter that accelerates SGD in the relevant direction and dampens oscillations.
        /// </summary>
        /// <value>
        /// The momentum.
        /// </value>
        public float Momentum { get; set; }

        /// <summary>
        /// Learning rate decay over each update.
        /// </summary>
        /// <value>
        /// The decay rate.
        /// </value>
        public float DecayRate { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="BaseOptimizer"/> class.
        /// </summary>
        /// <param name="lr">The lr.</param>
        /// <param name="name">The name.</param>
        public BaseOptimizer(float lr, string name)
        {
            LearningRate = lr;
            Name = name;
        }

        /// <summary>
        /// Updates the specified iteration.
        /// </summary>
        /// <param name="iteration">The iteration.</param>
        /// <param name="layer">The layer.</param>
        public abstract void Update(int iteration, BaseLayer layer);

        /// <summary>
        /// Gets the specified optimizer type.
        /// </summary>
        /// <param name="optimizerType">Type of the optimizer.</param>
        /// <returns></returns>
        public static BaseOptimizer Get(string name)
        {
            BaseOptimizer opt = null;
            switch (name)
            {
                case "sgd":
                    break;
                case "adam":
                    opt = new Adam();
                    break;
                default:
                    break;
            }

            return opt;
        }
    }
    /// <summary>
    /// Adam is an optimization algorithm that can used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.
    /// <para>
    /// Adam was presented by Diederik Kingma from OpenAI and Jimmy Ba from the University of Toronto in their 2015 ICLR paper(poster) titled “Adam: A Method for Stochastic Optimization“. 
    /// I will quote liberally from their paper in this post, unless stated otherwise.
    /// </para>
    /// </summary>
    public class Adam : BaseOptimizer
    {
        /// <summary>
        /// Gets or sets the beta 1 value.
        /// </summary>
        /// <value>
        /// The beta1.
        /// </value>
        public float Beta1 { get; set; }

        /// <summary>
        /// Gets or sets the beta 2 value.
        /// </summary>
        /// <value>
        /// The beta2.
        /// </value>
        public float Beta2 { get; set; }

        private Dictionary<string, NDArray> ms;
        private Dictionary<string, NDArray> vs;

        public Adam(float lr = 0.01f, float beta_1 = 0.9f, float beta_2 = 0.999f, float decayRate = 0) : base(lr, "adam")
        {
            Beta1 = beta_1;
            Beta2 = beta_2;
            DecayRate = decayRate;
            ms = new Dictionary<string, NDArray>();
            vs = new Dictionary<string, NDArray>();
        }

        public override void Update(int iteration, BaseLayer layer)
        {
            //If Decay rate is more than 0, the correct the learnng rate per iteration.
            if (DecayRate > 0)
            {
                LearningRate = LearningRate * (1 / (1 + DecayRate * iteration));
            }

            //Loop through all the parameters in the layer
            foreach (var p in layer.Parameters.ToList())
            {
                //Get the parameter name
                string paramName = p.Key;

                //Create a unique name to store in the dictionary
                string varName = layer.Name + "_" + p.Key;

                //Get the weight values
                NDArray param = p.Value;

                //Get the gradient/partial derivative values
                NDArray grad = layer.Grads[paramName];

                //If this is first time, initlalise all the moving average values with 0
                if (!ms.ContainsKey(varName))
                {
                    //ToDo: np.full
                    //var ms_new = Constant(0, param.shape);
                    //ms[varName] = ms_new;
                }

                //If this is first time, initlalise all the moving average values with 0
                if (!vs.ContainsKey(varName))
                {
                    //ToDo: np.full
                    //var vs_new = Constant(0, param.Shape);
                    //vs[varName] = vs_new;
                }

                // Calculate the exponential moving average for Beta 1 against the gradient
                ms[varName] = (Beta1 * ms[varName]) + (1 - Beta1) * grad;

                //Calculate the exponential squared moving average for Beta 2 against the gradient
                vs[varName] = (Beta2 * vs[varName]) + (1 - Beta2) * np.power(grad, 2);

                //Correct the moving averages
                var m_cap = ms[varName] / (1 - (float)Math.Pow(Beta1, iteration));
                var v_cap = vs[varName] / (1 - (float)Math.Pow(Beta2, iteration));

                //Update the weight of of the neurons
                layer.Parameters[paramName] = param - (LearningRate * m_cap / (np.sqrt(v_cap) + Epsilon));
            }
        }
    }
    #endregion

    #region NeuralNet
    /// <summary>
    /// Sequential model builder with train and predict
    /// </summary>
    public class NeuralNet
    {
        public event EventHandler<EpochEndEventArgs> EpochEnd;

        /// <summary>
        /// Layers which the model will contain
        /// </summary>
        public List<BaseLayer> Layers { get; set; }

        /// <summary>
        /// The optimizer instance used during training
        /// </summary>
        public BaseOptimizer Optimizer { get; set; }

        /// <summary>
        /// The cost instance for the training
        /// </summary>
        public BaseCost Cost { get; set; }

        /// <summary>
        /// The metric instance for the training
        /// </summary>
        public BaseMetric Metric { get; set; }

        /// <summary>
        /// Training losses for all the iterations
        /// </summary>
        public List<float> TrainingLoss { get; set; }

        /// <summary>
        /// Training metrices for all the iterations
        /// </summary>
        public List<float> TrainingMetrics { get; set; }

        /// <summary>
        /// Create instance of the neural net with parameters
        /// </summary>
        /// <param name="optimizer"></param>
        /// <param name="cost"></param>
        /// <param name="metric"></param>
        public NeuralNet(BaseOptimizer optimizer, BaseCost cost, BaseMetric metric = null)
        {
            Layers = new List<BaseLayer>();
            TrainingLoss = new List<float>();
            TrainingMetrics = new List<float>();

            this.Optimizer = optimizer != null ? optimizer : throw new Exception("Need optimizer");
            this.Cost = cost != null ? cost : throw new Exception("Need cost");
            Metric = metric;
        }

        /// <summary>
        /// Helper method to stack layer
        /// </summary>
        /// <param name="layer"></param>
        public void Add(BaseLayer layer)
        {
            Layers.Add(layer);
        }
       
        /// <summary>
        /// Train the model with training dataset, for certain number of iterations and using batch size
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="numIterations"></param>
        /// <param name="batchSize"></param>
        public void Train(NDArray x, NDArray y, int numIterations, int batchSize)
        {
            
            //Initialise bacch loss and metric list for temporary holding of result
            List<float> batchLoss = new List<float>();
            List<float> batchMetrics = new List<float>();

            Stopwatch sw = new Stopwatch();

            //Loop through till the end of specified iterations
            for (int i = 1; i <= numIterations; i++)
            {
                sw.Start();

                //Initialize local variables
                int currentIndex = 0;
                batchLoss.Clear();
                batchMetrics.Clear();

                //Loop untill the data is exhauted for every batch selected
                while (true)
                {
                    //Get the batch data based on the specified batch size
                    var xtrain = x[currentIndex].reshape(1,x.shape[1]);//,currentIndex+batchSize];
                    var ytrain = y[currentIndex].reshape(1,y.shape[1]);//,currentIndex+batchSize];//, currentIndex + y.shape[1]];
                    
                    if (xtrain == null)
                        break;

                    //Run forward for all the layers to predict the value for the training set
                    var ypred = Forward(xtrain);

                    //Find the loss/cost value for the prediction wrt expected result
                    var costVal = Cost.Forward(ypred, ytrain);
                    batchLoss.AddRange(costVal.Data<float>());

                    //Find the metric value for the prediction wrt expected result
                    if (Metric != null)
                    {
                        var metric = Metric.Calculate(ypred, ytrain);
                        batchMetrics.AddRange(metric.Data<float>());
                    }

                    //Get the gradient of the cost function which is the passed to the layers during back-propagation
                    var grad = Cost.Backward(ypred, ytrain);

                    //Run back-propagation accross all the layers
                    Backward(grad);

                    //Now time to update the neural network weights using the specified optimizer function
                    foreach (var layer in Layers)
                    {
                        Optimizer.Update(i, layer);
                    }

                    currentIndex = currentIndex++;// batchSize; 
                    if (currentIndex >= x.shape[0]) break;
                }

                sw.Stop();
                //Collect the result and fire the event
                float batchLossAvg = (float)Math.Round(batchLoss.Average(), 2);

                float batchMetricAvg = Metric != null ? (float)Math.Round(batchMetrics.Average(), 2) : 0;

                TrainingLoss.Add(batchLossAvg);

                if (batchMetrics.Count > 0)
                    TrainingMetrics.Add(batchMetricAvg);

                EpochEndEventArgs eventArgs = new EpochEndEventArgs(i, batchLossAvg, batchMetricAvg, sw.ElapsedMilliseconds);
                EpochEnd?.Invoke(i, eventArgs);
                sw.Reset();
            }
        }

        /// <summary>
        /// Prediction method
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public NDArray Predict(NDArray x)
        {
            return Forward(x);
        }

        /// <summary>
        /// Internal method to execute forward method accross all the layers
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        private NDArray Forward(NDArray x)
        {
            BaseLayer lastLayer = null;

            foreach (var layer in Layers)
            {
                if (lastLayer == null)
                    layer.Forward(x);
                else
                    layer.Forward(lastLayer.Output);

                lastLayer = layer;
            }

            return lastLayer.Output;
        }

        /// <summary>
        /// Internal method to execute back-propagation method accross all the layers
        /// </summary>
        /// <param name="gradOutput"></param>
        private void Backward(NDArray gradOutput)
        {
            var curGradOutput = gradOutput;
            for (int i = Layers.Count - 1; i >= 0; --i)
            {
                var layer = Layers[i];

                layer.Backward(curGradOutput);
                curGradOutput = layer.InputGrad;
            }
        }
    }

    public class EpochEndEventArgs
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="BatchEndEventArgs"/> class.
        /// </summary>
        /// <param name="epoch">The current epoch number.</param>
        /// <param name="batch">The current batch number.</param>
        /// <param name="loss">The loss value for the batch.</param>
        /// <param name="metric">The metric value for the batch.</param>
        public EpochEndEventArgs(
            int epoch,
            float loss,
            float metric,
            long duration)
        {
            Epoch = epoch;
            Loss = loss;
            Metric = metric;
            Duration = duration;
        }

        /// <summary>
        /// Gets the current epoch number.
        /// </summary>
        /// <value>
        /// The epoch.
        /// </value>
        public int Epoch { get; }

        /// <summary>
        /// Gets the loss value for this batch.
        /// </summary>
        /// <value>
        /// The loss.
        /// </value>
        public float Loss { get; }

        /// <summary>
        /// Gets the metric value for this batch.
        /// </summary>
        /// <value>
        /// The metric.
        /// </value>
        public float Metric { get; }

        /// <summary>
        /// Time taken in ms per iteration
        /// </summary>
        public long Duration { get; }
    }
    #endregion

    #region helpers
    public class Util
    {
        private static int counter = 0;

        public static int GetNext()
        {
            return counter++;
        }
    }
    #endregion

    public class Testing
    {
        static void Main()
        {
            //tarik data csv
            var datasetPath = $"{FileHelpers.AppDirectory}\\..\\..\\..\\..\\Dataset\\auto-mpg.csv";

            var data = DatasetHelper.LoadAsDataTable(datasetPath);
            //hapus kolom
            data.Drop(new[] { "car name" });
            //one hot encoding
            data.OneHotEncoding("origin");
            //split training n test data
            var (dt_train, dt_test) = data.Split();
            //lihat data contoh
            data.Head();
            //buang kolom y
            NDArray y_train = dt_train.Pop2("mpg");
            //normalisasi dengan z-score
            data.Normalization();
            //features
            NDArray x_train = dt_train.ToNDArray2();

            NeuralNet net = new NeuralNet(new Adam(), new MeanSquaredError(), new MeanAbsoluteError());
            net.Layers.Add(new FullyConnected(dt_train.Columns.Count, 32,"relu"));
            net.Layers.Add(new FullyConnected(32, 64,"relu"));
            net.Layers.Add(new FullyConnected(64, 1));
            net.EpochEnd += (_, e) => {
                Console.WriteLine($"-> epoch: {e.Epoch}, time: { new TimeSpan(e.Duration).TotalSeconds + " seconds"}, loss: {e.Loss}, MSE: {e.Metric}");
            };
           
            net.Train(x_train, y_train, 100, 1);
            Console.WriteLine("training completed.");
            Console.ReadLine();
        }

        
    }

}