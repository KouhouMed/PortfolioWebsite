+++
title = "Supervised Learning algorithms with R (part 2)"
description = "2nd part of the previous article about some of the most common supervised learning algorithms and their implementation with R"
author = "Mohamed Kouhou"
date = "2021-01-07"
tags = ["machine learning", "data science"]
categories = ["machine learning", "data science"]
[[images]]
  src = "https://i.ibb.co/8D9B0dC/supervised-learning-r-2.png"
  alt = "ml with r"
  stretch = "stretchH"
+++

In a <a href="https://kouhoumed.site/blog/supervised-learning-algorithms-r/">previous article</a>, we have explained the principles of K-Nearest Neighbours and Decision Trees and seen hands-on examples of how these models can be created using R programming language. In this article, we will continue on the same track and see other supervised learning algorithms, namely **Linear Regression** and **Logistic Regression**.  

**Regression** is one of the most important fields in statistics and machine learning. Its objective is to find relationships between variables. On use case of regression is when you try to find out how some features of employees -- such as experience, role, education and city -- influence their salaries. To solve this problem, we apply regression on a dataset comprised of a large enough amount observations, each representing the data (features) related to each employee. 

***When do we need Regression?*** 

Basically, regression is used to answer the following questions :
* Does X influence Y?
* If the answer is yes, how is Y influenced?
* In other words, how Y is related to each of the input features, and how these features are correlated?

Regression is almost always present in decision-making processes in all fields such as economy, demographic analysis, real estate industry, education, etc.

## Linear Regression

>**Linear Regression** is the simplest regression method and the most widely used because of the ease of interpreting results. It is a predictive model used to predict the outcome value of an outcome variable (**dependent variable**) Y based on one or more input predictor variables (**independent variables**) X. The aim is to establish a linear relationship (a mathematical formula) between the predictor variable(s) and the response variable, so that we can use this formula to estimate the value of the response Y, when only the predictors values are known.

A linear regression between dependent variable $Y$ and a set of independent variables $X=(x_1,...,x_n)$ assumes that there is a linear relationship between $X$ and $Y$, i.e. :
$$Y=\beta_0 + \beta_1x_1 + ... + \beta_nx_n + \varepsilon$$ 
The equation above is called the **regression equation**. $\beta_0, \beta_1, ..., \beta_n$ are called the **regression coefficients**, and $\varepsilon$ is the **random error**.

Linear regression calculates the **predicted weights** $\hat{\beta}_0, \hat{\beta}_1, ..., \hat{\beta}_n$ that define the **regression function** :
$$f(X)=\hat{\beta}_0 + \hat{\beta}_1x_1 + ... + \hat{\beta}_nx_n$$

The **estimated response** $f(X^{(i)})$ of each observation $i=1,...,N$ should be as close as possible to the corresponding **actual response** $Y^{(i)}$. The differences $Y^{(i)}-f(X^{(i)})$ for all the observations $i=1,...,N$ are called **residuals**. Regression is about determining the best predicted weights, that is the weights corresponding to the smallest residuals.  

In order to obtain the best weights, we try to **minimize the sum of squared residuals (SSR)** for all observations $i=1,...,N$ :
$$SSR=\sum_i (Y^{(i)}-f(X^{(i)}))^2$$
This method is called the method of **ordinary least squares**. 

<center>
  <figure>
    <img src="https://i1.wp.com/statisticsbyjim.com/wp-content/uploads/2017/04/residuals.png?resize=300%2C186&ssl=1"/>
    <figcaption>Residuals are the distance between the observed value and the fitted value.</figcaption>
  </figure>
</center>

After fitting our model, we would eventually want to measure how good the fit is. One way to do it is by calculating $R^2$ also called **[Coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination)**. 

The variation of the actual responses $Y^{(i)}$ is partly due to their dependence on $X$. But there is also a part of the overall variation that is intrinsic to the output itself.

$$R^2=\dfrac{\text{Variance explained by the model}}{\text{Total variance}}=1-\dfrac{SSR}{TSS}$$

$SSR$ : [sum of squared residuals](https://en.wikipedia.org/wiki/Residual_sum_of_squares)  
$TSS$ : [total sum of squares](https://en.wikipedia.org/wiki/Total_sum_of_squares)

$R^2$ is the amount of variation in $Y$ that can be explained by its relationship with $X$. Usually, the larger the $R^2$, the better the regression model fits your observations and can better explain the variation of the output with different inputs. $R^2=1$ corresponds to the case where $SSR=0$
, which is the **perfect fit**.

<center>
  <div class="inline-block">
    <figure>
      <img src ="https://i0.wp.com/statisticsbyjim.com/wp-content/uploads/2017/04/flp_highvar.png?resize=300%2C210&ssl=1">
      <figcaption>$R^2=15\%$</figcaption>
    </figure>
  </div>
  <div class="inline-block">
    <figure>
      <img src ="https://i2.wp.com/statisticsbyjim.com/wp-content/uploads/2017/04/flp_lowvar.png?resize=300%2C209&ssl=1">
      <figcaption>$R^2=85\%$</figcaption>
    </figure>
  </div>
</center>

<style>
  .inline-block {
    display: inline-block;
  }
</style>

#### Simple linear regression

**Simple Linear Regression** is the simplest case of linear regression as it only involves a single independent variable $X=x$.

<center>
  <figure>
    <img src="https://miro.medium.com/max/2584/0*Y_wKuvGOCaoUQKeJ.png" style="width: 50%;
    height: auto"/>
    <figcaption>Simple Linear Regression plot</figcaption>
  </figure>
</center>

Simple linear regression helps us summarize and study relationships between two continuous (quantitative) variables $X=x$ and $Y$.