+-----------------------+----------------------------------------------------+----------------------+
| **King Khalid         | ![](media/image2.svg){width="0.9577077865266842in" | **[جامعة الملك       |
| University**          | height="0.7583333333333333in"}                     | خالد]{dir="rtl"}**   |
|                       |                                                    |                      |
| **College of Computer |                                                    | **[كلية علوم         |
| Science**             |                                                    | الحاسب]{dir="rtl"}** |
|                       |                                                    |                      |
| **Department of       |                                                    | **[قسم علوم          |
| Computer Science**    |                                                    | الحاسب]{dir="rtl"}** |
+=======================+====================================================+======================+

**[STUDENT LAB ACTIVITY REPORT]{.underline}**

+----------------------------------+------------------------------------------+
| **Course Name:** Introduction to | **Lab Teacher: Dr. Aymen TRIGUI**        |
| Machine Learning                 |                                          |
|                                  | **Student ID:**                          |
| **Course Code:** 462 CCS-3       | .......................................  |
|                                  |                                          |
| **Section Number:                | **Student                                |
| [903]{.underline} S. No**        | Name:**................................. |
| ............                     |                                          |
+==================================+==========================================+

**[LAB ACTIVITY - 6]{.underline} (3 Marks)**

Write a Python program to demonstrate a comparative analysis between the
KNN algorithm and Logistic Regression. This activity should focus on
cross-validation and associate tools to select between the two models.

**Note that this dataset can also be loaded from the scikit_learn
library using the following Python instructions:**

**from** **sklearn.datasets** **import** load_iris

Activity Requirement:

1)  first load the Iris dataset and perform feature scaling
    using StandardScaler. Then, split the dataset into training and test
    sets using train_test_split.

2)  Create the KNN classifier with KNeighborsClassifier and the Logistic
    Regression classifier with LogisticRegression.

3)  Use cross_val_score to perform 5-fold cross-validation on both
    models and obtain the cross-validation scores.

4)  Finally, compare the mean cross-validation scores of the two models
    and select the model with the higher score as the better-performing
    model.
