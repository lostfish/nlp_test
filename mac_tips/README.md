# Mac Tips

## 1. 装机必备软件

+ [Alfred](https://www.alfredapp.com): 类似Spotlight的搜索工具，查文件，执行命令，写workflow
+ [iTerm2](https://www.iterm2.com): 更好用的终端，配合oh-my-zsh使用
+ [oh-my-zsh](https://ohmyz.sh): 加强版zsh，点开有安装命令
+ [Solarized](https://github.com/altercation/solarized): solarized配色方案
+ [CheetSheet](https://www.mediaatelier.com/CheatSheet): 快捷键提示，长按command键显示
+ [JieTu](http://jietu.qq.com): 截图工具,App Store安装
+ Magnet: 窗口管理工具,App Store安装
+ [Typora](https://www.typora.io): Markdown文件编辑器，可导出为pdf，word，html等
+ [Xmind8](https://www.xmind.cn/xmind8-pro/): 思维导图工具
+ [Homebrew](https://brew.sh): brew安装软件，点开有安装命令
+ [Parallels](https://www.parallels.com): Mac上运行windows虚拟器，试用版，比virtualbox好很多

other:
https://github.com/hzlzh/Best-App/blob/master/README.md

## 2. 使用mac遇到的问题汇总

### Issue 1: In virtualenv, import tensorflow in jupyter

ref: http://niranjan.co/installing-tensorflow-with-jupyter/

solve: install ipython & jupyter in virutalenv (note the order)

    pip install --ignore-installed scandir (maybe)
    pip install --ignore-installed ipython
    pip install --upgrade jupyter
    which ipython
    which jupyter

## Issue 2: Install rz & sz

ref: https://github.com/mmastrac/iterm2-zmodem
