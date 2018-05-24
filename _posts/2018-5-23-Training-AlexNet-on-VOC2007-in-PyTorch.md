---
layout: post
title: Training AlexNet on VOC2007 in PyTorch
published: true
---

The goal of this post is to introduce data loading and model training through PyTorch. We will use the VOC2007 dataset and Alexnet to train an image detection network.

Clone the tutorials repo:

{% highlight javascript %}
cd ~
git clone https://github.com/grudsby/tutorials

{% endhighlight %}

Download the VOC2007 dataset:

{% highlight javascript %}
cd ~/tutorials/alexnet-voc2007-pytorch
mkdir data
cd data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar && tar xf VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar && tar xf VOCtest_06-Nov-2007.tar
cd ../

{% endhighlight %}

Take a look at the VOC2007 dataset structure. 
