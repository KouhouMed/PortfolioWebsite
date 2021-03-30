+++
title = "Supervised Learning algorithms with R"
description = "We will discover some of the most common supervised learning algorithms and their implementation with R programming language"
author = "Mohamed Kouhou"
date = "2021-01-05"
tags = ["machine learning", "data science"]
categories = ["machine learning", "data science"]
[[images]]
  src = "https://i.ibb.co/rFW5Mc2/supervised-learning-r.png"
  alt = "ml"
  stretch = "stretchH"
+++

As I mentioned in an earlier <a href="https://kouhoumed.site/blog/ml/" >post</a>, machine learning algorithms are categorized into three main types :
* Supervised learning algorithms
* Unsupervised learning algorithms
* Reinfocement learning algorithms

In this article, we will only talk about some of machine learning supervised algorithms. We'll also see examples using R programming language.  

First of all, I would like to remind you that supervised learning is a machine learning algorithm that tries to find a function mapping an input to a given output based on a set of examples. Basically, each of these examples (called **training set**) consists of the input $X$ and the desired output $Y$. In other words, we try to approximate the function $f$ such that $f(X)=Y$ using previously labelled data as learning examples. The performance of such a model is evaluated upon its ability to generalize onto new data that is unlabeled. 

## K-Nearest Neighbours
K-Nearest Neighbours (KNN) is a simple Machine Learning algorithm based on Supervised Learning technique. It can be used for both regression and classification, but it is mostly used in classification. 

Suppose we have two categories : Category $A$ and Category $B$, and we have a new data point $x$ that we want to know to which category it belongs.
<center><img src="https://miro.medium.com/max/800/1*2zYNhLc522h0zftD1zDh2g.png" style="width: 50%;
  height: auto"/></center>

KNN algorithm is comprised of the following steps :
- **Step 1 :** Select K, number of neighbours
- **Step 2 :** Calculate the distance of between $x$ and each of the other data points (we can use the Euclidian distance or others)
- **Step 3 :** Take the K nearest neighbors as per the calculated Euclidean distance.
- **Step 4 :** Among these k neighbors, count the number of the data points in each category.
- **Step 5 :** Assign the new data points to that category for which the number of the neighbor is maximum.


**NB 1 :** The Euclidian distance between two points $a_1(x_1,y_1)$ and $a_2(x_2,y_2)$ is calculated as follows : $$d(a_1,a_2)=\sqrt{(x_1-x_2)^2+(y_1-y_2)^2}$$
<center><img src="https://lh3.googleusercontent.com/proxy/KkYtKow-RsWwJtXyFj89QLfreCXbi00NVtB88MrKEBN4gPcHMv2BrXJvVC6wgprf2J1CNAmATpv9rbIjDQw9P6DbOVg9JKeKnQ" style="width: 50%;
  height: auto"/></center>
A more general formula in an n-dimensional space is : $$d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}$$ where $x(x_1,...,x_n)$ and $y(y_1,...,y_n)$.

**NB 2 :** As can be seen, there are no parameters that need to be learned during training to determine whether a new observation belongs to class  𝐴  or  𝐵.  The only parameter used in k-nearest neighbours is K, which is a predetermined value. The algorithm simply works by looking at the training samples, calculating distances and finding the K examples in the training set that are closest to the new observation. Thus, KNN is a **non-parametric**, **supervised** (needs training labels) learning algorithm.

**NB 3 :** KNN does  support <a href="https://en.wikipedia.org/wiki/Categorical_variable">categorical variables</a> as features, simply because we cannot calculated the distance from them.
