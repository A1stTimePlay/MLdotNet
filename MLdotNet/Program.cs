using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;

namespace MLdotNet
{
    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _dataPath => Path.Combine(_appPath, "Data", "train.csv");
        private static string _modelPath => Path.Combine(_appPath, "Models", "model.zip");

        private static MLContext _mlContext;

        private static PredictionEngine<DataObject, PredictedObject> _predEngine;

        private static ITransformer _trainedModel;

        private static TrainTestData _dataView;

        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);

            _dataView = LoadData(_mlContext);

            var pipeline = ProcessData();

            _trainedModel = BuildAndTrainModel(_mlContext, _dataView.TrainSet, pipeline);

            _predEngine = BuildPredictionEngine(_mlContext, _trainedModel);

            Evaluate(_mlContext, _dataView.TestSet, _trainedModel);

            SaveModelAsFile(_mlContext, _dataView.TrainSet.Schema, _trainedModel);


        }

        public static TrainTestData LoadData(MLContext mLContext)
        {
            // load data, sau đó chia thành tập train test với tỉ lệ 80 train - 20 test

            Console.WriteLine($"=============== Loading Dataset  ===============");

            var dataView = _mlContext.Data.LoadFromTextFile<DataObject>(_dataPath, separatorChar: ',', hasHeader: true);
            TrainTestData splitDataView = mLContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            Console.WriteLine($"=============== Finished Loading Dataset  ===============");

            return splitDataView;
        }

        public static IEstimator<ITransformer> ProcessData() 
        {
            // featurize các cột chữ, sau đó thêm các cột đã featurize và các cột số thành cột features

            Console.WriteLine($"=============== Processing Data ===============");

            var pipeline = _mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "fraudulent")
                .Append(_mlContext.Transforms.Concatenate("Features", "telecommuting", "has_company_logo", "has_questions"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "title", outputColumnName: "titleFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "location", outputColumnName: "locationFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "department", outputColumnName: "departmentFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "company_profile", outputColumnName: "company_profileFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "description", outputColumnName: "descriptionFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "requirements", outputColumnName: "requirementsFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "benefits", outputColumnName: "benefitsFeaturized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "titleFeaturized", "locationFeaturized", "departmentFeaturized", "company_profileFeaturized", "descriptionFeaturized", "requirementsFeaturized", "benefitsFeaturized", "telecommuting", "has_company_logo", "has_questions"))
                ;

            Console.WriteLine($"=============== Finished Processing Data ===============");

            return pipeline;
        }

        public static ITransformer BuildAndTrainModel(MLContext mLContext, IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            // tạm thời sử dụng thuật toán SdcaLogisticRegression

            Console.WriteLine($"=============== Training Model ===============");

            var trainingPipeline = pipeline.Append(mLContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            // tới đây mới bắt đầu train model nè, ở trên chỉ là setting cho cái pipeline thôi. Hiện giờ đang bị lỗi, nghi vấn do dữ liệu không sạch
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            Console.WriteLine($"=============== Finished Training Model ===============");

            return trainedModel;
        }

        public static PredictionEngine<DataObject, PredictedObject> BuildPredictionEngine(MLContext mLContext, ITransformer trainedModel)
        {
            // tạo predict engine, sau này sử dụng để predict 1 hoặc nhiều giá trị nhập vào

            Console.WriteLine($"=============== Building Prediction Engine ===============");

            var preEngine = mLContext.Model.CreatePredictionEngine<DataObject, PredictedObject>(trainedModel);

            Console.WriteLine($"=============== Finished Building Prediction Engine ===============");
            return preEngine;
        }

        public static void Evaluate(MLContext mLContext, IDataView testingDataView, ITransformer trainedModel)
        {
            // show một số độ đo để kiểm nghiệm model đã xây dựng

            var predictions = trainedModel.Transform(testingDataView);
            var metrics = mLContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        private static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            // save model đã xây dựng vào file 

            mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
        }
    }
}
