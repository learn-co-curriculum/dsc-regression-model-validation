# Regression Model Validation

## Introduction

Previously you've evaluated a multiple linear regression model by calculating metrics based on the fit of the training data. In this lesson you'll learn why it's important to split your data in a train and a test set if you want to evaluate a model used for prediction.

## Objectives

You will be able to:

* Perform a train-test split
* Prepare training and testing data for modeling
* Compare training and testing errors to determine if model is over or underfitting

## Model Evaluation

Recall some ways that we can evaluate linear regression models.

### Residuals

It is pretty straightforward that, to evaluate the model, you'll want to compare your predicted values, $\hat y$ with the actual value, $y$. The difference between the two values is referred to as the **residuals**:

$r_{i} = y_{i} - \hat y_{i}$ 

To get a summarized measure over all the instances, a popular metric is the (Root) Mean Squared Error:

RMSE = $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_{i} - \hat y_{i})^2}$

MSE = $\frac{1}{n}\sum_{i=1}^{n}(y_{i} - \hat y_{i})^2$

Larger (R)MSE indicates a _worse_ model fit.

## The Need for Train-Test Split

### Making Predictions and Evaluation

So far we've simply been fitting models to data, and evaluated our models calculating the errors between our $\hat y$ and our actual targets $y$, while these targets $y$ contributed in fitting the model.

Let's say we want to predict the outcome for observations that are not necessarily in our dataset now; e.g: we want to **predict** miles per gallon for a new car that isn't part of our dataset, or predict the price for a new house in Ames.

In order to get a good sense of how well your model will be doing on new instances, you'll have to perform a so-called "train-test-split". What you'll be doing here, is taking a sample of the data that serves as input to "train" our model - fit a linear regression and compute the parameter estimates for our variables, and calculate how well our predictive performance is doing comparing the actual targets $y$ and the fitted $\hat y$ obtained by our model.

### Underfitting and Overfitting

Another reason to use train-test-split is because of a common problem which doesn't only affect linear models, but nearly all (other) machine learning algorithms: overfitting and underfitting. An overfit model is not generalizable and will not hold to future cases. An underfit model does not make full use of the information available and produces weaker predictions than is feasible. The following image gives a nice, more general demonstration:

<img src='https://curriculum-content.s3.amazonaws.com/data-science/images/new_overfit_underfit.png'>

## Mechanics of Train-Test Split

When performing a train-test-split, it is important that the data is **randomly** split. At some point, you will encounter datasets that have certain characteristics that are only present in certain segments of the data. For example, if you were looking at sales data for a website, you might expect the data to look different on days that promotional deals were held versus days that deals were not held. If we don't randomly split the data, there is a chance we might overfit to the characteristics of certain segments of data.

Another thing to consider is just **how big** each training and testing set should be. There is no hard and fast rule for deciding the correct size, but the range of training set is usually anywhere from 66% - 80% (and testing set between 33% and 20%). Some types of machine learning models need a substantial amount of data to train on, and as such, the training sets should be larger. Some models with many different tuning parameters will need to be validated with larger sets (the test size should be larger) to determine what the optimal parameters should be. When in doubt, just stick with training set sizes around 70% and test set sizes around 30%.

## Train-Test Split with Scikit-Learn

You could write your own pandas code to shuffle and split your data, but we'll use the convenient `train_test_split` function from scikit-learn instead. We'll also use the Auto MPG dataset.


```python
import pandas as pd

data = pd.read_csv('auto-mpg.csv')
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model year</th>
      <th>origin</th>
      <th>car name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>387</th>
      <td>27.0</td>
      <td>4</td>
      <td>140.0</td>
      <td>86</td>
      <td>2790</td>
      <td>15.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford mustang gl</td>
    </tr>
    <tr>
      <th>388</th>
      <td>44.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>52</td>
      <td>2130</td>
      <td>24.6</td>
      <td>82</td>
      <td>2</td>
      <td>vw pickup</td>
    </tr>
    <tr>
      <th>389</th>
      <td>32.0</td>
      <td>4</td>
      <td>135.0</td>
      <td>84</td>
      <td>2295</td>
      <td>11.6</td>
      <td>82</td>
      <td>1</td>
      <td>dodge rampage</td>
    </tr>
    <tr>
      <th>390</th>
      <td>28.0</td>
      <td>4</td>
      <td>120.0</td>
      <td>79</td>
      <td>2625</td>
      <td>18.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford ranger</td>
    </tr>
    <tr>
      <th>391</th>
      <td>31.0</td>
      <td>4</td>
      <td>119.0</td>
      <td>82</td>
      <td>2720</td>
      <td>19.4</td>
      <td>82</td>
      <td>1</td>
      <td>chevy s-10</td>
    </tr>
  </tbody>
</table>
<p>392 rows × 9 columns</p>
</div>



The `train_test_split` function ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)) takes in a series of array-like variables, as well as some optional arguments. It returns multiple arrays.

For example, this would be a valid way to use `train_test_split`:


```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(data)
```


```python
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model year</th>
      <th>origin</th>
      <th>car name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>217</th>
      <td>25.5</td>
      <td>4</td>
      <td>122.0</td>
      <td>96</td>
      <td>2300</td>
      <td>15.5</td>
      <td>77</td>
      <td>1</td>
      <td>plymouth arrow gs</td>
    </tr>
    <tr>
      <th>211</th>
      <td>13.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>145</td>
      <td>4055</td>
      <td>12.0</td>
      <td>76</td>
      <td>1</td>
      <td>chevy c10</td>
    </tr>
    <tr>
      <th>60</th>
      <td>21.0</td>
      <td>4</td>
      <td>122.0</td>
      <td>86</td>
      <td>2226</td>
      <td>16.5</td>
      <td>72</td>
      <td>1</td>
      <td>ford pinto runabout</td>
    </tr>
    <tr>
      <th>203</th>
      <td>28.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>75</td>
      <td>2155</td>
      <td>16.4</td>
      <td>76</td>
      <td>3</td>
      <td>toyota corolla</td>
    </tr>
    <tr>
      <th>198</th>
      <td>18.0</td>
      <td>6</td>
      <td>250.0</td>
      <td>78</td>
      <td>3574</td>
      <td>21.0</td>
      <td>76</td>
      <td>1</td>
      <td>ford granada ghia</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>298</th>
      <td>23.9</td>
      <td>8</td>
      <td>260.0</td>
      <td>90</td>
      <td>3420</td>
      <td>22.2</td>
      <td>79</td>
      <td>1</td>
      <td>oldsmobile cutlass salon brougham</td>
    </tr>
    <tr>
      <th>175</th>
      <td>23.0</td>
      <td>4</td>
      <td>115.0</td>
      <td>95</td>
      <td>2694</td>
      <td>15.0</td>
      <td>75</td>
      <td>2</td>
      <td>audi 100ls</td>
    </tr>
    <tr>
      <th>355</th>
      <td>30.7</td>
      <td>6</td>
      <td>145.0</td>
      <td>76</td>
      <td>3160</td>
      <td>19.6</td>
      <td>81</td>
      <td>2</td>
      <td>volvo diesel</td>
    </tr>
    <tr>
      <th>93</th>
      <td>13.0</td>
      <td>8</td>
      <td>440.0</td>
      <td>215</td>
      <td>4735</td>
      <td>11.0</td>
      <td>73</td>
      <td>1</td>
      <td>chrysler new yorker brougham</td>
    </tr>
    <tr>
      <th>271</th>
      <td>23.9</td>
      <td>4</td>
      <td>119.0</td>
      <td>97</td>
      <td>2405</td>
      <td>14.9</td>
      <td>78</td>
      <td>3</td>
      <td>datsun 200-sx</td>
    </tr>
  </tbody>
</table>
<p>294 rows × 9 columns</p>
</div>




```python
test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model year</th>
      <th>origin</th>
      <th>car name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>135</th>
      <td>13.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>150</td>
      <td>4699</td>
      <td>14.5</td>
      <td>74</td>
      <td>1</td>
      <td>buick century luxus (sw)</td>
    </tr>
    <tr>
      <th>209</th>
      <td>16.5</td>
      <td>6</td>
      <td>168.0</td>
      <td>120</td>
      <td>3820</td>
      <td>16.7</td>
      <td>76</td>
      <td>2</td>
      <td>mercedes-benz 280s</td>
    </tr>
    <tr>
      <th>275</th>
      <td>16.2</td>
      <td>6</td>
      <td>163.0</td>
      <td>133</td>
      <td>3410</td>
      <td>15.8</td>
      <td>78</td>
      <td>2</td>
      <td>peugeot 604sl</td>
    </tr>
    <tr>
      <th>125</th>
      <td>19.0</td>
      <td>6</td>
      <td>232.0</td>
      <td>100</td>
      <td>2901</td>
      <td>16.0</td>
      <td>74</td>
      <td>1</td>
      <td>amc hornet</td>
    </tr>
    <tr>
      <th>71</th>
      <td>15.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3892</td>
      <td>12.5</td>
      <td>72</td>
      <td>1</td>
      <td>amc matador (sw)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>322</th>
      <td>40.8</td>
      <td>4</td>
      <td>85.0</td>
      <td>65</td>
      <td>2110</td>
      <td>19.2</td>
      <td>80</td>
      <td>3</td>
      <td>datsun 210</td>
    </tr>
    <tr>
      <th>82</th>
      <td>28.0</td>
      <td>4</td>
      <td>98.0</td>
      <td>80</td>
      <td>2164</td>
      <td>15.0</td>
      <td>72</td>
      <td>1</td>
      <td>dodge colt (sw)</td>
    </tr>
    <tr>
      <th>65</th>
      <td>17.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3672</td>
      <td>11.5</td>
      <td>72</td>
      <td>1</td>
      <td>amc ambassador sst</td>
    </tr>
    <tr>
      <th>137</th>
      <td>14.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>4638</td>
      <td>16.0</td>
      <td>74</td>
      <td>1</td>
      <td>ford gran torino (sw)</td>
    </tr>
    <tr>
      <th>385</th>
      <td>36.0</td>
      <td>4</td>
      <td>135.0</td>
      <td>84</td>
      <td>2370</td>
      <td>13.0</td>
      <td>82</td>
      <td>1</td>
      <td>dodge charger 2.2</td>
    </tr>
  </tbody>
</table>
<p>98 rows × 9 columns</p>
</div>



In this case, the DataFrame `data` was split into two DataFrames called `train` and `test`. `train` has 294 values (75% of the full dataset) and `test` has 98 values (25% of the full dataset). Note the randomized order of the index values on the left.

However you can also pass multiple array-like variables into `train_test_split` at once. For each variable that you pass in, you will get a train and a test copy back out.

Most commonly in this curriculum these are the inputs and outputs:

Inputs

- `X`
- `y`

Outputs

- `X_train`
- `X_test`
- `y_train`
- `y_test`


```python
y = data[['mpg']]
X = data.drop(['mpg', 'car name'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y)
```


```python
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model year</th>
      <th>origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>294</th>
      <td>4</td>
      <td>121.0</td>
      <td>80</td>
      <td>2670</td>
      <td>15.0</td>
      <td>79</td>
      <td>1</td>
    </tr>
    <tr>
      <th>240</th>
      <td>4</td>
      <td>121.0</td>
      <td>110</td>
      <td>2600</td>
      <td>12.8</td>
      <td>77</td>
      <td>2</td>
    </tr>
    <tr>
      <th>242</th>
      <td>4</td>
      <td>90.0</td>
      <td>48</td>
      <td>1985</td>
      <td>21.5</td>
      <td>78</td>
      <td>2</td>
    </tr>
    <tr>
      <th>39</th>
      <td>8</td>
      <td>351.0</td>
      <td>153</td>
      <td>4154</td>
      <td>13.5</td>
      <td>71</td>
      <td>1</td>
    </tr>
    <tr>
      <th>125</th>
      <td>6</td>
      <td>232.0</td>
      <td>100</td>
      <td>2901</td>
      <td>16.0</td>
      <td>74</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>59</th>
      <td>4</td>
      <td>140.0</td>
      <td>90</td>
      <td>2408</td>
      <td>19.5</td>
      <td>72</td>
      <td>1</td>
    </tr>
    <tr>
      <th>137</th>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>4638</td>
      <td>16.0</td>
      <td>74</td>
      <td>1</td>
    </tr>
    <tr>
      <th>369</th>
      <td>4</td>
      <td>105.0</td>
      <td>74</td>
      <td>1980</td>
      <td>15.3</td>
      <td>82</td>
      <td>2</td>
    </tr>
    <tr>
      <th>174</th>
      <td>6</td>
      <td>232.0</td>
      <td>90</td>
      <td>3211</td>
      <td>17.0</td>
      <td>75</td>
      <td>1</td>
    </tr>
    <tr>
      <th>198</th>
      <td>6</td>
      <td>250.0</td>
      <td>78</td>
      <td>3574</td>
      <td>21.0</td>
      <td>76</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>294 rows × 7 columns</p>
</div>




```python
y_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>294</th>
      <td>27.4</td>
    </tr>
    <tr>
      <th>240</th>
      <td>21.5</td>
    </tr>
    <tr>
      <th>242</th>
      <td>43.1</td>
    </tr>
    <tr>
      <th>39</th>
      <td>14.0</td>
    </tr>
    <tr>
      <th>125</th>
      <td>19.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>59</th>
      <td>20.0</td>
    </tr>
    <tr>
      <th>137</th>
      <td>14.0</td>
    </tr>
    <tr>
      <th>369</th>
      <td>36.0</td>
    </tr>
    <tr>
      <th>174</th>
      <td>19.0</td>
    </tr>
    <tr>
      <th>198</th>
      <td>18.0</td>
    </tr>
  </tbody>
</table>
<p>294 rows × 1 columns</p>
</div>



We can view the lengths of the results like this:


```python
print(len(X_train), len(X_test), len(y_train), len(y_test))
```

    294 98 294 98


However it is not recommended to pass in just the data to be split. This is because the randomization of the split means that you will get different results for `X_train` etc. every time you run the code. **For reproducibility, it is always recommended that you specify a `random_state`**, such as in this example:


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

Another optional argument is `test_size`, which makes it possible to choose the size of the test set and the training set instead of using the default 75% train/25% test proportions.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Note that the lengths of the resulting datasets will be different:


```python
print(len(X_train), len(X_test), len(y_train), len(y_test))
```

    313 79 313 79


## Preparing Data for Modeling

When using a train-test split, data preparation should happen _after_ the split. This is to avoid ***data leakage***. The general idea is that the treatment of the test data should be as similar as possible to how genuinely unknown data should be treated. And genuinely unknown data would not have been there at the time of fitting the scikit-learn transformers, just like it would not have been there at the time of fitting the model!

In some cases you will see all of the data being prepared together for expediency, but the best practice is to prepare it separately.

### Log Transformation


```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Instantiate a custom transformer for log transformation 
log_transformer = FunctionTransformer(np.log, validate=True)

# Columns to be log transformed 
log_columns = ['displacement', 'horsepower', 'weight']

# New names for columns after transformation
new_log_columns = ['log_disp', 'log_hp', 'log_wt']

# Log transform the training columns and convert them into a DataFrame 
X_train_log = pd.DataFrame(log_transformer.fit_transform(X_train[log_columns]), 
                           columns=new_log_columns, index=X_train.index)

# Replace training columns with transformed versions
X_train = pd.concat([X_train.drop(log_columns, axis=1), X_train_log], axis=1)
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cylinders</th>
      <th>acceleration</th>
      <th>model year</th>
      <th>origin</th>
      <th>log_disp</th>
      <th>log_hp</th>
      <th>log_wt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>258</th>
      <td>6</td>
      <td>18.7</td>
      <td>78</td>
      <td>1</td>
      <td>5.416100</td>
      <td>4.700480</td>
      <td>8.194229</td>
    </tr>
    <tr>
      <th>182</th>
      <td>4</td>
      <td>14.9</td>
      <td>76</td>
      <td>1</td>
      <td>4.941642</td>
      <td>4.521789</td>
      <td>7.852439</td>
    </tr>
    <tr>
      <th>172</th>
      <td>6</td>
      <td>14.5</td>
      <td>75</td>
      <td>1</td>
      <td>5.141664</td>
      <td>4.574711</td>
      <td>8.001020</td>
    </tr>
    <tr>
      <th>63</th>
      <td>8</td>
      <td>13.5</td>
      <td>72</td>
      <td>1</td>
      <td>5.762051</td>
      <td>5.010635</td>
      <td>8.327243</td>
    </tr>
    <tr>
      <th>340</th>
      <td>4</td>
      <td>16.4</td>
      <td>81</td>
      <td>1</td>
      <td>4.454347</td>
      <td>4.158883</td>
      <td>7.536364</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>71</th>
      <td>8</td>
      <td>12.5</td>
      <td>72</td>
      <td>1</td>
      <td>5.717028</td>
      <td>5.010635</td>
      <td>8.266678</td>
    </tr>
    <tr>
      <th>106</th>
      <td>6</td>
      <td>15.0</td>
      <td>73</td>
      <td>1</td>
      <td>5.446737</td>
      <td>4.605170</td>
      <td>7.933438</td>
    </tr>
    <tr>
      <th>270</th>
      <td>4</td>
      <td>17.6</td>
      <td>78</td>
      <td>1</td>
      <td>5.017280</td>
      <td>4.442651</td>
      <td>7.956827</td>
    </tr>
    <tr>
      <th>348</th>
      <td>4</td>
      <td>20.7</td>
      <td>81</td>
      <td>1</td>
      <td>4.584967</td>
      <td>4.174387</td>
      <td>7.774856</td>
    </tr>
    <tr>
      <th>102</th>
      <td>8</td>
      <td>14.0</td>
      <td>73</td>
      <td>1</td>
      <td>5.991465</td>
      <td>5.010635</td>
      <td>8.516593</td>
    </tr>
  </tbody>
</table>
<p>313 rows × 7 columns</p>
</div>




```python
# Log transform the test columns and convert them into a DataFrame 
X_test_log = pd.DataFrame(log_transformer.transform(X_test[log_columns]), 
                          columns=new_log_columns, index=X_test.index)

# Replace testing columns with transformed versions
X_test = pd.concat([X_test.drop(log_columns, axis=1), X_test_log], axis=1)
X_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cylinders</th>
      <th>acceleration</th>
      <th>model year</th>
      <th>origin</th>
      <th>log_disp</th>
      <th>log_hp</th>
      <th>log_wt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>78</th>
      <td>4</td>
      <td>18.0</td>
      <td>72</td>
      <td>2</td>
      <td>4.564348</td>
      <td>4.234107</td>
      <td>7.691200</td>
    </tr>
    <tr>
      <th>274</th>
      <td>4</td>
      <td>15.7</td>
      <td>78</td>
      <td>2</td>
      <td>4.795791</td>
      <td>4.744932</td>
      <td>7.935587</td>
    </tr>
    <tr>
      <th>246</th>
      <td>4</td>
      <td>16.4</td>
      <td>78</td>
      <td>3</td>
      <td>4.510860</td>
      <td>4.094345</td>
      <td>7.495542</td>
    </tr>
    <tr>
      <th>55</th>
      <td>4</td>
      <td>20.5</td>
      <td>71</td>
      <td>1</td>
      <td>4.510860</td>
      <td>4.248495</td>
      <td>7.578145</td>
    </tr>
    <tr>
      <th>387</th>
      <td>4</td>
      <td>15.6</td>
      <td>82</td>
      <td>1</td>
      <td>4.941642</td>
      <td>4.454347</td>
      <td>7.933797</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>361</th>
      <td>6</td>
      <td>16.6</td>
      <td>81</td>
      <td>1</td>
      <td>5.416100</td>
      <td>4.442651</td>
      <td>8.150468</td>
    </tr>
    <tr>
      <th>82</th>
      <td>4</td>
      <td>15.0</td>
      <td>72</td>
      <td>1</td>
      <td>4.584967</td>
      <td>4.382027</td>
      <td>7.679714</td>
    </tr>
    <tr>
      <th>114</th>
      <td>8</td>
      <td>13.0</td>
      <td>73</td>
      <td>1</td>
      <td>5.857933</td>
      <td>4.976734</td>
      <td>8.314342</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>5.717028</td>
      <td>5.010635</td>
      <td>8.141190</td>
    </tr>
    <tr>
      <th>18</th>
      <td>4</td>
      <td>14.5</td>
      <td>70</td>
      <td>3</td>
      <td>4.574711</td>
      <td>4.477337</td>
      <td>7.663877</td>
    </tr>
  </tbody>
</table>
<p>79 rows × 7 columns</p>
</div>



### One-Hot Encoding


```python
from sklearn.preprocessing import OneHotEncoder

# Instantiate OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Create X_cat which contains only the categorical variables
cat_columns = [col for col in X.columns if X[col].dtype in [object]]
X_train_cat = X_train.loc[:, cat_columns]

# Fill missing values with the string 'missing'
X_train_cat.fillna(value='missing', inplace=True)

# Transform training set
X_train_ohe = pd.DataFrame(ohe.fit_transform(X_train_cat),
                           columns=cat_columns, index=X_train.index)
```


```python
# Transform testing set
X_test_cat = pd.DataFrame(ohe.transform(X_test[cat_columns]),
                           columns=cat_columns, index=X_test.index)

# Fill missing values with the string 'missing'
X_test_cat.fillna(value='missing', inplace=True)

# Transform training set
X_test_ohe = pd.DataFrame(ohe.fit_transform(X_test_cat),
                           columns=cat_columns, index=X_test.index)
```

## Building, Evaluating, and Validating a Model

Great, now that you have preprocessed all the columns, you can fit a linear regression model: 


```python
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)
```

Look at the residuals and calculate the MSE for training and test sets:  


```python
train_residuals = y_hat_train - y_train
test_residuals = y_hat_test - y_test
```


```python
mse_train = np.sum((y_train - y_hat_train)**2)/len(y_train)
mse_test = np.sum((y_test - y_hat_test)**2)/len(y_test)
print('Train Mean Squared Error:', mse_train)
print('Test Mean Squared Error:', mse_test)
```

    Train Mean Squared Error: mpg    9.143784
    dtype: float64
    Test Mean Squared Error: mpg    10.010717
    dtype: float64


You can also do this directly using sklearn's `mean_squared_error()` function: 


```python
from sklearn.metrics import mean_squared_error

train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
print('Train Mean Squared Error:', train_mse)
print('Test Mean Squared Error:', test_mse)
```

    Train Mean Squared Error: 9.143783702580818
    Test Mean Squared Error: 10.010717292450195


Great, there does not seem to be a big difference between the train and test MSE! Interestingly, the test set error is smaller than the training set error. This is fairly rare but does occasionally happen.

In other words, our validation process has indicated that we are **not** overfitting. In fact, we may be _underfitting_ because linear regression is not a very complex model.

## Overfitting with a Different Model

Just for the sake of example, here is a model that is overfit to the data. Don't worry about the model algorithm being shown! Instead, just look at the MSE for the train vs. test set, using the same preprocessed data:


```python
from sklearn.tree import DecisionTreeRegressor

other_model = DecisionTreeRegressor(random_state=42)
other_model.fit(X_train, y_train)

other_train_mse = mean_squared_error(y_train, other_model.predict(X_train))
other_test_mse = mean_squared_error(y_test, other_model.predict(X_test))
print('Train Mean Squared Error:', other_train_mse)
print('Test Mean Squared Error:', other_test_mse)
```

    Train Mean Squared Error: 0.0
    Test Mean Squared Error: 10.84253164556962


This model initially seems great...0 MSE for the training data! But then you see that it is performing much worse than our linear regression model on the test data. This model **is** overfitting.

## Additional Resources

[This blog post](https://towardsdatascience.com/linear-regression-in-python-9a1f5f000606) shows a walkthrough of the key steps for model validation with train-test split and scikit-learn.

## Summary 

In this lesson, you learned the importance of the train-test split approach and used one of the most popular metrics for evaluating regression models, (R)MSE. You also saw how to use the `train_test_split` function from `sklearn` to split your data into training and test sets, and then evaluated whether models were overfitting using metrics on those training and test sets.
