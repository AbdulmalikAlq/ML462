+-----------------------------------------------------------------------+
| **Activity-1**                                                        |
|                                                                       |
| **Name: S. No:**                                                      |
|                                                                       |
| **Academic No: Section:**                                             |
|                                                                       |
| **Subject: 462 CCS**                                                  |
+=======================================================================+

**EXERCISE NO**: **1**

**TITLE OF EXERCISE**: **Scikit- Learn**

+-----------------------------------------------------------------------+
| **Goal AIM/OBJECTIVE:** This activity provides the Scikit Learn       |
| fundamentals.                                                         |
+=======================================================================+
| Write the Output                                                      |
|                                                                       |
| 1\.                                                                   |
|                                                                       |
| \# scipy                                                              |
|                                                                       |
| import scipy                                                          |
|                                                                       |
| print(\'scipy: {}\'.format(scipy.\_\_version\_\_))                    |
|                                                                       |
| \# numpy                                                              |
|                                                                       |
| import numpy                                                          |
|                                                                       |
| print(\'numpy: {}\'.format(numpy.\_\_version\_\_))                    |
|                                                                       |
| \# matplotlib                                                         |
|                                                                       |
| import matplotlib                                                     |
|                                                                       |
| print(\'matplotlib: {}\'.format(matplotlib.\_\_version\_\_))          |
|                                                                       |
| \# pandas                                                             |
|                                                                       |
| import pandas                                                         |
|                                                                       |
| print(\'pandas: {}\'.format(pandas.\_\_version\_\_))                  |
|                                                                       |
| 2\. **Accessing data using numpy**                                    |
|                                                                       |
| \# access values                                                      |
|                                                                       |
| import numpy                                                          |
|                                                                       |
| mylist = \[\[1, 2, 3\], \[3, 4, 5\]\]                                 |
|                                                                       |
| myarray = numpy.array(mylist)                                         |
|                                                                       |
| print(myarray)                                                        |
|                                                                       |
| print(myarray.shape)                                                  |
|                                                                       |
| print(\"First row: %s\" % myarray\[0\])                               |
|                                                                       |
| print(\"Last row: %s\" % myarray\[-1\])                               |
|                                                                       |
| print(\"Specific row and col: %s\" % myarray\[0, 2\])                 |
|                                                                       |
| print(\"Whole col: %s\" % myarray\[:, 2\] )                           |
|                                                                       |
| **3. Draw the Line plot**                                             |
|                                                                       |
| \# basic line plot                                                    |
|                                                                       |
| import matplotlib.pyplot as plt                                       |
|                                                                       |
| import numpy                                                          |
|                                                                       |
| myarray = numpy.array(\[1, 2, 3\])                                    |
|                                                                       |
| plt.plot(myarray)                                                     |
|                                                                       |
| plt.xlabel(\'some x axis\')                                           |
|                                                                       |
| plt.ylabel(\'some y axis\')                                           |
|                                                                       |
| plt.show()                                                            |
|                                                                       |
| **4.Draw Scatter Plot**                                               |
|                                                                       |
| \# basic scatter plot                                                 |
|                                                                       |
| import matplotlib.pyplot as plt                                       |
|                                                                       |
| import numpy                                                          |
|                                                                       |
| x = numpy.array(\[1, 2, 3\])                                          |
|                                                                       |
| y = numpy.array(\[2, 4, 6\])                                          |
|                                                                       |
| plt.scatter(x,y)                                                      |
|                                                                       |
| plt.xlabel(\'some x axis\')                                           |
|                                                                       |
| plt.ylabel(\'some y axis\')                                           |
|                                                                       |
| plt.show()                                                            |
|                                                                       |
| **5. DataFrame**                                                      |
|                                                                       |
| **\# dataframe**                                                      |
|                                                                       |
| import numpy                                                          |
|                                                                       |
| import pandas                                                         |
|                                                                       |
| myarray = numpy.array(\[\[1, 2, 3\], \[4, 5, 6\]\])                   |
|                                                                       |
| rownames = \[\'a\', \'b\'\]                                           |
|                                                                       |
| colnames = \[\'one\', \'two\', \'three\'\]                            |
|                                                                       |
| mydataframe = pandas.DataFrame(myarray, index=rownames,               |
| columns=colnames)                                                     |
|                                                                       |
| print(mydataframe)                                                    |
|                                                                       |
| **6. To load the iris data from a given csv file into a dataframe and |
| print the shape of the data, type of the data and first 3 rows**      |
|                                                                       |
| import pandas as pd                                                   |
|                                                                       |
| data = pd.read_csv(\"iris.csv\")                                      |
|                                                                       |
| print(\"Shape of the data:\")                                         |
|                                                                       |
| print(data.shape)                                                     |
|                                                                       |
| print(\"\\nData Type:\")                                              |
|                                                                       |
| print(type(data))                                                     |
|                                                                       |
| print(\"\\nFirst 3 rows:\")                                           |
|                                                                       |
| print(data.head(3))                                                   |
+-----------------------------------------------------------------------+
