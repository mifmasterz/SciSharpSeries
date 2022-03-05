using System.Data;
using Numpy;
using System.Linq;
using System.Text;

namespace ML.Tools
{
    public class DatasetHelper
    {
        public static DataTable LoadAsDataTable(string CsvPath, bool HasHeader = true, char Separator = ',', bool DropNa = true, string NaValue = "?")
        {
            if (!File.Exists(CsvPath)) throw new Exception("File tidak ada.");
            DataTable dataTable = new DataTable("data");
            var rowCount = 0;
            var rows = File.ReadAllLines(CsvPath);
            foreach (var line in rows)
            {
                var splitted = line.Split(Separator);
                if (rowCount == 0)
                    if (HasHeader)
                    {
                        foreach (var column in splitted)
                        {
                            dataTable.Columns.Add(column);
                        }
                    }
                    else
                    {
                        for (int i = 0; i < splitted.Length; i++)
                        {
                            dataTable.Columns.Add("Col" + i);
                        }
                    }
                else
                {
                    var newRow = dataTable.NewRow();
                    for (int i = 0; i < splitted.Length; i++)
                    {
                        newRow[i] = splitted[i];
                        if (splitted[i].ToString().Trim() == NaValue && DropNa)
                        {
                            goto loncat;
                        }
                    }
                    dataTable.Rows.Add(newRow);
                }
            loncat:

                rowCount++;
            }
            dataTable.AcceptChanges();
            return dataTable;
        }
    }
    public static partial class DatatableExtensions
    {
        public static void Drop(this DataTable dt, string[] ColumnNames)
        {
            var cols = new List<string>();
            foreach (DataColumn dc in dt.Columns)
            {
                cols.Add(dc.ColumnName);

            }
            foreach (var dc in ColumnNames)
            {
                if (cols.Contains(dc))
                {
                    dt.Columns.Remove(dc);

                }
            }
            dt.AcceptChanges();
        }
        
        public static (DataTable training, DataTable test) Split(this DataTable trainingTable, float ValidationPercent=0.2f)
        {
            //var trainingTable = dt.Clone();
            var take = (int)(ValidationPercent * trainingTable.Rows.Count);
            var testTable = trainingTable.Clone();
            testTable.Rows.Clear();
            Random rnd = new Random(Environment.TickCount);
            for (var i= 0; i < take;i++)
            {
                var rowSel = rnd.Next(0, trainingTable.Rows.Count-1);
                var newRow = testTable.NewRow();
                foreach (DataColumn column in trainingTable.Columns)
                {
                   newRow[column.ColumnName] = trainingTable.Rows[rowSel][column.ColumnName];
                }
                testTable.Rows.Add(newRow);
                trainingTable.Rows.RemoveAt(rowSel);
            }
            testTable.AcceptChanges();
            trainingTable.AcceptChanges();
            return (trainingTable,testTable);


        }
        /// <summary>
        /// Generates a textual representation of <paramref name="table"/>.
        /// </summary>
        /// <param name="table">The table to print.</param>
        /// <returns>A textual representation of <paramref name="table"/>.</returns>
        public static void Head(this DataTable table, int RowCount = 20)
        {
            String GetCellValueAsString(DataRow row, DataColumn column)
            {
                var cellValue = row[column];
                var cellValueAsString = cellValue is null or DBNull ? "{{null}}" : cellValue.ToString();

                return cellValueAsString;
            }

            var columnWidths = new Dictionary<DataColumn, Int32>();

            foreach (DataColumn column in table.Columns)
            {
                columnWidths.Add(column, column.ColumnName.Length);
            }
            var RowIndex = 0;
            foreach (DataRow row in table.Rows)
            {
                foreach (DataColumn column in table.Columns)
                {
                    columnWidths[column] = Math.Max(columnWidths[column], GetCellValueAsString(row, column).Length);
                }
                RowIndex++;
                if (RowIndex >= RowCount) break;
            }

            var resultBuilder = new StringBuilder();

            resultBuilder.Append("| ");

            foreach (DataColumn column in table.Columns)
            {
                resultBuilder.Append(column.ColumnName.PadRight(columnWidths[column]));
                resultBuilder.Append(" | ");
            }

            resultBuilder.AppendLine();

            foreach (DataRow row in table.Rows)
            {
                resultBuilder.Append("| ");

                foreach (DataColumn column in table.Columns)
                {
                    resultBuilder.Append(GetCellValueAsString(row, column).PadRight(columnWidths[column]));
                    resultBuilder.Append(" | ");
                }

                resultBuilder.AppendLine();
            }
            Console.WriteLine(resultBuilder.ToString());
            //return resultBuilder.ToString();
        }

        public static void Normalization(this DataTable dt)
        {
            var values = new List<double>();
            double sum = 0f;
            double mean = 0f;
            double bigSum = 0;
            foreach (DataColumn dc in dt.Columns)
            {
                foreach (DataRow dr in dt.Rows)
                {
                    values.Add(double.Parse(dr[dc.ColumnName].ToString()));
                }
                //calculate
                sum = 0f;
                mean = 0f;
                bigSum = 0;
                foreach (var value in values)
                    sum += value;
                mean = sum / values.Count;

                // Calculate the total for the standard deviation
                for (int i = 0; i < values.Count; i++)
                    bigSum += Math.Pow(values[i] - mean, 2);

                // Now we can calculate the standard deviation
                var stdDev = Math.Sqrt(bigSum / (values.Count - 1));


                for (var i = 0; i < values.Count; i++)
                {
                    var value = values[i];
                    values[i] = (value - mean) / stdDev;
                    dt.Rows[i][dc.ColumnName] = values[i];
                }
                values.Clear();
            }
            dt.AcceptChanges();
        }
        public static void OneHotEncoding(this DataTable dt, string ColumnName)
        {
            var AllDatas = new List<string>();
            var cols = new List<string>();
            foreach (DataColumn dc in dt.Columns)
            {
                cols.Add(dc.ColumnName);

            }

            if (cols.Contains(ColumnName))
            {
                foreach (DataRow dr in dt.Rows)
                {
                    AllDatas.Add(dr[ColumnName].ToString().Trim());
                }
                AllDatas = AllDatas.Distinct().ToList();
                var newCols = new List<string>();
                AllDatas.ForEach(x => newCols.Add($"{ColumnName}_{x}"));
                foreach (var col in newCols)
                {
                    dt.Columns.Add(col);
                }
                foreach (DataRow dr in dt.Rows)
                {
                    for (var idx = 0; idx < newCols.Count; idx++)
                    {
                        var col = newCols[idx];
                        dr[col] = AllDatas[idx] == dr[ColumnName].ToString().Trim() ? 1 : 0;
                    }
                }
                dt.Columns.Remove(ColumnName);

            }

            dt.AcceptChanges();
        }
        public static NDarray GetByRowIndex(this DataTable dt, int Index)
        {
            if (Index < 0 || Index >= dt.Rows.Count) throw new ArgumentOutOfRangeException("index");
            var floats = new List<float>();
            foreach (DataColumn dc in dt.Columns)
            {

                floats.Add(float.Parse(dt.Rows[Index][dc.ColumnName].ToString()));

            }
            return np.array(floats.ToArray());
        }
        public static NDarray Pop(this DataTable dt, string ColumnName)
        {
            //var floats = new List<float>();
            var floats = new float[dt.Rows.Count,1];
            var rowIndex = 0;
            foreach (DataColumn dc in dt.Columns)
            {
                if (ColumnName == dc.ColumnName)
                {
                    foreach (DataRow dr in dt.Rows)
                    {
                        floats[rowIndex,0] = (float.Parse(dr[ColumnName].ToString()));
                        rowIndex++;
                    }
                }
            }
            dt.Columns.Remove(ColumnName);
            dt.AcceptChanges();
            return np.array(floats);
        } 
        
        public static string[] ToStringArray(this DataTable dt, string ColumnName)
        {
            //var floats = new List<float>();
            var values = new List<string>();
            var rowIndex = 0;
            foreach (DataColumn dc in dt.Columns)
            {
                if (ColumnName == dc.ColumnName)
                {
                    foreach (DataRow dr in dt.Rows)
                    {
                        values.Add (dr[ColumnName].ToString());
                        rowIndex++;
                    }
                }
            }
            return values.ToArray();
        }

        public static string[] ToCategory(this DataTable dt, string ColumnName)
        {
            //var floats = new List<float>();
            var values = new List<string>();
            var rowIndex = 0;
            var temp = (from x in dt.AsEnumerable()
                      select x).ToList();
            values = temp.AsEnumerable().Select(x=>x.Field<string>(ColumnName)).Distinct().ToList();
            foreach(DataRow dataRow in dt.Rows)
            {
                var index = Array.FindIndex(values.ToArray(), row => row== dataRow[ColumnName].ToString().Trim());
                dataRow[ColumnName] = index;
            }
            dt.AcceptChanges();
            return values.ToArray();
        }

        public static NDarray ToNDArray(this DataTable dt)
        {
            var floats = new float[dt.Rows.Count, dt.Columns.Count];
            var rowIndex = 0;
            var colIndex = 0;

            foreach (DataRow dr in dt.Rows)
            {
                colIndex = 0;
                foreach (DataColumn dc in dt.Columns)
                {
                    floats[rowIndex, colIndex] = float.Parse(dr[dc.ColumnName].ToString());
                    colIndex++;
                }
                rowIndex++;
            }

            return np.array(floats);
        }
    }
}
