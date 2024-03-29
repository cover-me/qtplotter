# qtplotter

Note: The initial goal of this project was to generate both interactive and static figures with the same Jupyter Notebook code. However, the focus gradually shifts to generating static figures since the demand from publication is higher than sharing the data with a convenient visualization tool. For interactive figures, I started a new project [qtview](https://github.com/cover-me/qtview). Old source codes are still available on the release page of this project. 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cover-me/qtplotter/master?filepath=index.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cover-me/qtplotter/blob/master/index.ipynb)

A data visualization notebook inspired by [Rubenknex/qtplot](https://github.com/Rubenknex/qtplot) and [gsteele13/spyview](https://github.com/gsteele13/spyview). Most of the code is grabbed from qtplot.

It's a simpler, easier-to-access version of qtplot for visualizing and interacting with our shared quantum transport data.

## Files

index.ipynb: A demo for interactively visualizing shared-online data. 

example.ipynb: More examples, especially of generating static figures.

qtplotter.py: The main code is here.


## Preview .ipynb files

These files can be directly previewed on GitHub. GitHub doesn't render ipywidgets (which is used for interaction). [nbviewer.jupyter.org](https://nbviewer.jupyter.org/) may provide better previews. (Links: [index.ipynb](https://nbviewer.jupyter.org/github/cover-me/qtplotter/blob/master/index.ipynb), [example.ipynb](https://nbviewer.jupyter.org/github/cover-me/qtplotter/blob/master/example.ipynb)). However, even Jupyter nbviewer can not render complicated widgets correctly at this moment. Many widgets just disappear in all notebook viewers (a quick check: export notebook as an HTML file and check if anything is missing). As notebooks are used for sharing and viewing, it's important to avoid displaying images in a widget if this problem has not been solved. And here is how:

Run `Player.PLAYER_STATIC = True` once and all "interactive" figures generated by `play()` will be just static snapshots that can be directly saved in a notebook for sharing/viewing. Remember to comment out or delete `Player.PLAYER_STATIC = True` before saving/sharing the notebook.

## Run .ipynb files in the cloud

Click the badges at the begging of this readme file to launch index.ipynb.

### Binder

The notebooks can be run on [mybinder.org](https://mybinder.org/) directly so that the shared data, no matter local or online, can be visualized interactively with a few clicks, by anyone, anywhere. 

It usually takes about 40 seconds to launch, sometimes less than 20 s, sometimes more than 1 minute. After having launched, use the `open->files...` menu to go to the files page where you can open other notebooks or download generated files.

### Colab

It also works on Colab. Colab has native support for table of contents, which is awesome for notebooks. Colab open notebooks very fast. No one wants to wait for opening a notebook.

The downside of Colab is that it doesn't support ipywidgets very well. No supoort for `%matplotlib widget` at all, but `%matplotlib inline` mode works (even for interaction, though a bit slower than `widget` as `inline` backend redraws the whole figure every time). Everything in a Tab widget goes to the first Tab page, but that is not a scientific issue.

Colab doesn't import additional files from GitHub repositories automatically as Binder does. One has to upload qtplotter.py to Colab's temporary runtime manually to run the notebook, or copy-paste-run code from qtplotter.py in the notebook. The correct "uploading file" menu is on the left of the Colab interface. There is a "folder" icon below the "show table of contents" icon and the "show code snippet pane" icon. Uploading files to google drive also works but requires some hacking. Another way to dump files from this repositories is to run the code below in the notebook:

```
!git clone https://github.com/cover-me/qtplotter
%cd "qtplotter"
```
