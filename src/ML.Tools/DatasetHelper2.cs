using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NumSharp;

namespace ML.Tools
{
    public static partial class DatatableExtensions
    {
     
       
        public static NDArray GetByRowIndex2(this DataTable dt, int Index)
        {
            if (Index < 0 || Index >= dt.Rows.Count) throw new ArgumentOutOfRangeException("index");
            var floats = new List<float>();
            foreach (DataColumn dc in dt.Columns)
            {

                floats.Add(float.Parse(dt.Rows[Index][dc.ColumnName].ToString()));

            }
            return np.array(floats.ToArray());
        }
        public static NDArray Pop2(this DataTable dt, string ColumnName)
        {
            //var floats = new List<float>();
            var floats = new float[dt.Rows.Count, 1];
            var rowIndex = 0;
            foreach (DataColumn dc in dt.Columns)
            {
                if (ColumnName == dc.ColumnName)
                {
                    foreach (DataRow dr in dt.Rows)
                    {
                        floats[rowIndex, 0] = (float.Parse(dr[ColumnName].ToString()));
                        rowIndex++;
                    }
                }
            }
            dt.Columns.Remove(ColumnName);
            dt.AcceptChanges();
            return np.array(floats);
        }

      

        public static NDArray ToNDArray2(this DataTable dt)
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
