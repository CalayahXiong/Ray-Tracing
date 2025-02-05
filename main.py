import my_scene_parser
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pathlib

parse = argparse.ArgumentParser()
parse.add_argument("-i", "--infile", nargs='+', type=str, default='scenes/Sphere.json', help="Name of json file that will define the scene")#, required = True)
parse.add_argument("-o", "--outdir", type=str, default="out", help="directory for output files")
parse.add_argument('-s', '--show', action='store_true', help="Show the final image in a window")
parse.add_argument('-f', '--factor', type=float, default=1.0, help="Scale factor for resolution")

args = parse.parse_args()

if __name__ == "__main__":
    for f in args.infile:
        full_scene = my_scene_parser.load_scene(f)  # using my_scene_parser here if want to compute scene with added features
        # features
        full_scene.width = int(full_scene.width * args.factor)
        full_scene.height = int(full_scene.height * args.factor)
        image = full_scene.render()
        # remove the path and extension from scene file, put it in outdir with png extension
        outdir = pathlib.Path(args.outdir)
        outdir.mkdir(exist_ok=True) # Create output directory if it doesn't exist
        fout = str(outdir / pathlib.Path(f).stem) + ".png"
        print("Saving image to", fout)
        matplotlib.image.imsave(fout, image, vmin=0, vmax=1)
        if ( args.show ):
            plt.axis("off")
            plt.imshow(image)
            plt.show()