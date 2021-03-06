import numpy as np
import os
from pda import model_path, logger

class visualization(object):

    def __init__(self):
        pass
    def create_sprite(self, images):
        """
        Creates a sprite of provided images.

        :param images: Images to construct the sprite.
        :type images: `np.array`
        :return: An image array containing the sprite.
        :rtype: `np.ndarray`
        """

        shape = np.shape(images)

        if len(shape) < 3 or len(shape) > 4:
            raise ValueError('Images provided for sprite have wrong dimensions ' + str(len(shape)))

        if len(shape) == 3:
            # Check to see if it's mnist type of images and add axis to show image is gray-scale
            images = np.expand_dims(images, axis=3)
            shape = np.shape(images)

        # Change black and white images to RGB
        if shape[3] == 1:
            images = self.convert_to_rgb(images)

        n = int(np.ceil(np.sqrt(images.shape[0])))
        padding = ((0, n ** 2 - images.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (images.ndim - 3)
        images = np.pad(images, padding, mode='constant', constant_values=0)

        # Tile the individual thumbnails into an image
        images = images.reshape((n, n) + images.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, images.ndim + 1)))
        images = images.reshape((n * images.shape[1], n * images.shape[3]) + images.shape[4:])
        sprite = (images * 255).astype(np.uint8)

        return sprite

    def convert_to_rgb(self, images):
        """
        Converts grayscale images to RGB. It changes NxHxWx1 to a NxHxWx3 array, where N is the number of figures,
        H is the high and W the width.

        :param images: Grayscale images of shape (NxHxWx1).
        :type images: `np.ndarray`
        :return: Images in RGB format of shape (NxHxWx3).
        :rtype: `np.ndarray`
        """
        dims = np.shape(images)
        if not ((len(dims) == 4 and dims[-1] == 1) or len(dims) == 3):
            raise ValueError('Unexpected shape for grayscale images:' + str(dims))

        if dims[-1] == 1:
            # Squeeze channel axis if it exists
            rgb_images = np.squeeze(images, axis=-1)
        else:
            rgb_images = images
        rgb_images = np.stack((rgb_images,) * 3, axis=-1)

        return rgb_images

    def save_image(self, image, f_name):
        """
        Saves image into a file inside `DATA_PATH` with the name `f_name`.

        :param image: Image to be saved
        :type image: `np.ndarray`
        :param f_name: File name containing extension e.g., my_img.jpg, my_img.png, my_images/my_img.png
        :type f_name: `str`
        :return: `None`
        """
        file_name = os.path.join(model_path, f_name)
        folder = os.path.split(file_name)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        from PIL import Image
        im = Image.fromarray(image)
        im.save(file_name)
        logger.info('Image saved to %s.', file_name)

    def plot_3d(self, points, labels, colors=None, save=True, f_name=''):
        """
        Generates a 3-D plot in of the provided points where the labels define the
        color that will be used to color each data point.
        Concretely, the color of points[i] is defined by colors(labels[i]).
        Thus, there should be as many labels as colors.

        :param points: arrays with 3-D coordinates of the plots to be plotted
        :type points: `np.ndarray`
        :param labels: array of integers that determines the color used in the plot for the data point.
            Need to start from 0 and be sequential from there on.
        :type labels: `lst`
        :param colors: Optional argument to specify colors to be used in the plot. If provided, this array should contain
        as many colors as labels.
        :type `lst`
        :param save:  When set to True, saves image into a file inside `DATA_PATH` with the name `f_name`.
        :type save: `bool`
        :param f_name: Name used to save the file when save is set to True
        :type f_name: `str`
        :return: fig
        :rtype: `matplotlib.figure.Figure`
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from mpl_toolkits import mplot3d

            if colors is None:
                colors = []
                for i in range(len(np.unique(labels))):
                    colors.append('C' + str(i))
            else:
                if len(colors) != len(np.unique(labels)):
                    raise ValueError('The amount of provided colors should match the number of labels in the 3pd plot.')

            fig = plt.figure()
            ax = plt.axes(projection='3d')

            for i, coord in enumerate(points):
                try:
                    color_point = labels[i]
                    ax.scatter3D(coord[0], coord[1], coord[2], color=colors[color_point])
                except IndexError:
                    raise ValueError('Labels outside the range. Should start from zero and be sequential there after')
            if save:
                file_name = os.path.realpath(os.path.join(model_path, f_name))
                folder = os.path.split(file_name)[0]

                if not os.path.exists(folder):
                    os.makedirs(folder)
                fig.savefig(file_name, bbox_inches='tight')
                logger.info('3d-plot saved to %s.', file_name)

            return fig
        except ImportError:
            logger.warning("matplotlib not installed. For this reason, cluster visualization was not displayed.")


def plot_2d(points, labels, colors=None, save=True, f_name=''):
    """
    Generates a 3-D plot in of the provided points where the labels define the
    color that will be used to color each data point.
    Concretely, the color of points[i] is defined by colors(labels[i]).
    Thus, there should be as many labels as colors.

    :param points: arrays with 2-D coordinates of the plots to be plotted
    :type points: `np.ndarray`
    :param labels: array of integers that determines the color used in the plot for the data point.
        Need to start from 0 and be sequential from there on.
    :type labels: `lst`
    :param colors: Optional argument to specify colors to be used in the plot. If provided, this array should contain
    as many colors as labels.
    :type `lst`
    :param save:  When set to True, saves image into a file inside `DATA_PATH` with the name `f_name`.
    :type save: `bool`
    :param f_name: Name used to save the file when save is set to True
    :type f_name: `str`
    :return: fig
    :rtype: `matplotlib.figure.Figure`
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d

        if colors is None:
            colors = []
            for i in range(len(np.unique(labels))):
                colors.append('C' + str(i))
        else:
            if len(colors) != len(np.unique(labels)):
                raise ValueError('The amount of provided colors should match the number of labels in the 2-d plot.')

        fig = plt.figure()
        ax = plt.axes()

        for i, coord in enumerate(points):
            try:
                color_point = labels[i]
                ax.scatter(coord[0], coord[1], color=colors[color_point])
            except IndexError:
                raise ValueError('Labels outside the range. Should start from zero and be sequential there after')
        if save:
            file_name = os.path.realpath(os.path.join(model_path, f_name))
            folder = os.path.split(file_name)[0]

            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig(file_name, bbox_inches='tight')
            logger.info('2d-plot saved to %s.', file_name)

        return fig
    except ImportError:
        logger.warning("matplotlib not installed. For this reason, cluster visualization was not displayed.")

def plot_VAT(X):
    """
    Test of cluster tendency

    Installation:
    import rpy2.robjects.packages as rpackages
    utils = rpackages.importr('utils')
    utils.install_packages("FactoMineR")
    utils.install_packages("factoextra")
    """

    from rpy2.robjects.packages import importr

    import rpy2.robjects.packages as rpackages
    utils = rpackages.importr('utils')
    utils.install_packages("factoextra")

    ########
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    factoextra=rpackages.importr("factoextra")
    import rpy2.robjects as robjects
    grad = robjects.r('list(low = "black", high = "white")')
    clustend = factoextra.get_clust_tendency(X, 100, gradient=grad)
    print(clustend)
    pass
