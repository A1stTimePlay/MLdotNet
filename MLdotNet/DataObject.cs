using Microsoft.ML.Data;

namespace MLdotNet
{
    public class DataObject 
    {
        // tạm thời lấy 11 cột này, 10 cột đầu làm feature, cột fraudulent là label
        [LoadColumn(0)]
        public string title { get; set; }
        [LoadColumn(1)]
        public string location { get; set; }
        [LoadColumn(2)]
        public string department { get; set; }
        [LoadColumn(3)]
        public string company_profile { get; set; }
        [LoadColumn(4)]
        public string description { get; set; }
        [LoadColumn(5)]
        public string requirements { get; set; }
        [LoadColumn(6)]
        public string benefits { get; set; }
        [LoadColumn(7)]
        public float telecommuting { get; set; }
        [LoadColumn(8)]
        public float has_company_logo { get; set; }
        [LoadColumn(9)]
        public float has_questions { get; set; }
        [LoadColumn(10)]
        public bool fraudulent { get; set; }
    }

    public class PredictedObject
    {
        [ColumnName("PredictedLabel")]
        public bool fraudulent { get; set; }
    }
}
