import matplotlib.pyplot as plt


def heatmap(matrix, ax, cmap="YlOrRd", write_text=True, fontsize=7, valfunc=None, colorfunc=None):
    """Plots the given matrix as a heatmap on the given axes.

    Args: 
        matrix: A 2D numpy array that consists of either int or float datatypes.
        ax: matplotlib axes
        cmap (str): color palette of the heatmap
        write_text: If true, writes the value of the entries of the matrix into
        the heatmap.
        fontsize: Font size of the text in the entries of the heatmap.
        valfunc: Function that formats the values of the heatmap before displaying
        it as text.
        colorfunc: Function that takes in the value of the entry and outputs the
        color corresponding to the text of its entry in the heatmap.


    Returns:
        Plots the matrix on the axes using ``imshow``, then returns this image.
    """
    im = ax.imshow(matrix, cmap=cmap)
    valfunc = valfunc if valfunc is not None else lambda x : x
    colorfunc = colorfunc is colorfunc is not None else lambda x: "black"
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            val = matrix[i,j]
            if write_text:
                im.axes.text(
                    j,
                    i,
                    valfunc(val),
                    fontsize=fontsize,
                    ha="center",
                    va="center",
                    color=colorfunc(val)
                )
    return im
