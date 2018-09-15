---
layout: post
title: Ubuntu下手动安装codeblocks
author: Pekary
date: 2017-10-26-13:00:00 +0800
tags: Linux
---


# Ubuntu 下手动安装Codeblocks

1. 首先去官网下载源码文件 [下载地址](http://www.codeblocks.org/downloads/25)

2. 解压

   ```shell
   tar -zxvf codeblocks_16.01.tar.gz -C ~/
   ```

3. 编译配置

   安装过程中可能会出现如下错误：

   ```shell
   Makefile:372: recipe for target 'aclocal.m4' failed
   ```

   解决办法：

   ```shell
   sudo apt install autoreconf automake
   ```

   然后执行下面命令：

   ```shell
   sudo autoconf -f -i
   sudo mkdir /usr/local/codeblocks
   sudo ~/codeblocks_16.01/configure --prefix=/usr/local/codeblocks
   sudo make
   sudo make install
   ```

4. 如果想要在终端执行codeblocks

   ```shell
   sudo ln -s /usr/local/codeblocks /usr/bin/codeblocks
   ```



`在写这篇博客的时候遇到了一个非常有意思的事情，如果新博客发布时间在当前时间以后，这篇博客是不会显示在运行的网站上的`