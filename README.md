# qtplotter

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cover-me/qtplotter/master?filepath=index.ipynb)

A data visualization notebook inspired by [Rubenknex/qtplot](https://github.com/Rubenknex/qtplot) and [gsteele13/spyview](https://github.com/gsteele13/spyview). Most of the code is grabbed from qtplot.

It's a simpler, easier-to-access version of qtplot for visualizing and interacting with our shared quantum transport data.

## Files

index.ipynb: A demo for interactively visualizing shared-online data. 

example.ipynb: More examples, especially of generating static figures.

qtplotter.py: The main code is here.


## Preview .ipynb files

These files can be directly previewed on GitHub. However, GitHub doesn't render ipywidgets (which is used for interaction). You can preview notebooks on [jupyter.org](https://nbviewer.jupyter.org/) for a better experience. (Links: [index.ipynb](https://nbviewer.jupyter.org/github/cover-me/qtplotter/blob/master/index.ipynb), [example.ipynb](https://nbviewer.jupyter.org/github/cover-me/qtplotter/blob/master/example.ipynb))

## Run .ipynb files in the cloud

The notebooks can be run on [mybinder.org](https://mybinder.org/) directly so that the shared data, no matter local or online, can be visualized interactively with a few clicks, by anyone, anywhere. 

Tips: click the "launch binder" badge to launch the index.ipynb on mybinder.org. It usually takes less than 20 seconds to launch, but sometimes more than 1 minute. After having launched, use the `open->files...` menu to go to the files page where you can open other notebooks or download generated files.

It also works in Colab, which is very convenient if you have a google account. Colab doesn't support ipywidgets (or %matplotlib widget) very well. But %matplotlib inline mode also works for interaction.
