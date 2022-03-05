using NumSharp;
using NumSharpDemo;

//mengubah dimensi array jadi 2 axis: 3 dan 5
var a = np.arange(15).reshape(3, 5);
a?.Print("arrange and reshape");

//melihat ukuran tiap axis
NumsharpExtensions.Print("shape:");
NumsharpExtensions.Print(a.shape);

//jumlah dimensi
NumsharpExtensions.Print("ndim:");
NumsharpExtensions.Print(a.ndim);

//total item axis 1 x axis 2 x ... axis n
NumsharpExtensions.Print("size:");
NumsharpExtensions.Print(a.size);

//bikin 2 dimensi array dari numeric array
var b = np.array(new [,]{ {1.5, 2, 3}, { 4, 5, 6} });
b?.Print("create 2 dim array");

//bikin array isi 0 dengan parameter ukuran matrix
var c = np.zeros((3, 4));
c?.Print("create zero array");

//membuat array isi 1 dengan parameter ukuran matrix
var d = np.ones((3, 3));
d?.Print("create ones array");

//membuat array kosong dengan parameter ukuran matrix
var e = np.empty((2, 3));
e?.Print("create empty array");

//membuat deret berdasarkan rentang dan jumlah deret
var f = np.linspace(0, 2, 9);
f?.Print("linspace 0 - 2 : 9");

//aplikasi rumus dengan linspace
var g = np.linspace(0, 180, 10);
var h = np.sin(g);
g?.Print("derajat");
h?.Print("fungsi sin dari rentang derajat hasil linspace");

//bikin array dengan nilai acak
var rnd = np.random;
a = rnd.rand((2, 2));
a?.Print("buat array dengan nilai acak");

//operasi aritmatika
b = a + 10;
b?.Print("a ditambah 10");
c = a * 10;
c?.Print("a dikali 10");

//operasi matrix
d = np.array(new[,] { { 1, 1 } });
d?.Print("d");
e = np.array(new[,] { { 0, 1 },{ 2,3 } });
e?.Print("e");
f = d.dot(e);
f?.Print("perkalian matrix d . e");


//min, max
g = rnd.rand((3,3));
g?.Print("g");
g.min().Print("min");
g.max().Print("max");

//universal function
b = np.arange(8);
b.Print("b");
c = np.exp(b);
c?.Print("exp b");
//akar
d = np.sqrt(b);
d?.Print("sqrt b");


//indexing
a = np.arange(10);
a?.Print("a");
//ambil array ke 3 sampe 5
b = a["2:5"];
b?.Print("ambil array ke 3 sampe 5");

//flatten, sum, mean, std, iterasi
c = np.array(new [,] { { 1, 2 }, { 3, 4 } });
c?.Print("c");
// versi flat array dari a
d = c.flat; //or nd.flatten() for a copy
d?.Print("flatten c");
Console.WriteLine("iterasi");
// interate ndarray
foreach (object val in c)
{
    Console.WriteLine(val);
}
e = d.sum();
e?.Print("sum");
d.mean().Print("mean");
d.std().Print("std");

Console.ReadLine();