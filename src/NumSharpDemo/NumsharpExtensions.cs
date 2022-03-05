using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NumSharpDemo
{
    public static partial class NumsharpExtensions
    {
        static Random rnd = new Random();
        static ConsoleColor[] Colors;
        public static void Print(object data)
        {
            SetColors();

            Console.ForegroundColor = Colors[rnd.Next(0, Colors.Length)];

            Console.WriteLine(data.ToString());
        }
            public static void Print(this NDArray data)
        {
            SetColors();
           
            Console.ForegroundColor = Colors[rnd.Next(0, Colors.Length)];

            Console.WriteLine(data.ToString());
        }
        
        public static void Print(this NDArray data,string Label)
        {
            SetColors();
           
            Console.ForegroundColor = Colors[rnd.Next(0, Colors.Length)];

            Console.WriteLine($"{Label}:\n{data.ToString()}");
        }

       static void SetColors()
        {
            if (Colors == null)
            {
                var colors = new List<ConsoleColor>();
                foreach (ConsoleColor color in Enum.GetValues(typeof(ConsoleColor)))
                {
                    colors.Add(color);
                }
                Colors = colors.ToArray();
            }
        }
    }
}
