import argparse
import pickle
import tensorflow as tf
import numpy as np
from npde_helper import build_model, fit_model, load_model, save_model

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def train(args):
    with tf.Session() as sess:
        t = load_pickle(args.t_file)
        Y = load_pickle(args.Y_file)
        
        npde = build_model(sess, t, Y, model=args.model, sf0=args.sf0, ell0=args.ell0, 
                           W=args.W, ktype=args.ktype, whiten=args.whiten)
        
        npde = fit_model(sess, npde, t, Y, num_iter=args.num_iter, print_every=args.print_every, 
                         eta=args.eta, plot_=args.plot)
        
        save_model(npde, args.output_file)
        return "Model trained and saved to {}".format(args.output_file)

def predict(args):
    with tf.Session() as sess:
        npde = load_model(args.model_file, sess)
        x0 = load_pickle(args.x0_file)
        t = load_pickle(args.t_file)
        
        path = npde.predict(x0, t)
  
        return path
    
def main():
    parser = argparse.ArgumentParser(description='NPDE CLI')
    subparsers = parser.add_subparsers(dest='command')

    # Train parser
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--t_file', required=True, help='Pickle file containing t')
    train_parser.add_argument('--Y_file', required=True, help='Pickle file containing Y')
    train_parser.add_argument('--model', default='ode', choices=['ode', 'sde'], help='Model type')
    train_parser.add_argument('--sf0', type=float, default=1.0, help='Initial signal variance')
    train_parser.add_argument('--ell0', type=float, nargs='+', default=[1.0, 1.0], help='Initial lengthscale')
    train_parser.add_argument('--W', type=int, default=6, help='Width of inducing point grid')
    train_parser.add_argument('--ktype', default='id', help='Kernel type')
    train_parser.add_argument('--whiten', action='store_true', help='Whether to whiten')
    train_parser.add_argument('--num_iter', type=int, default=500, help='Number of iterations')
    train_parser.add_argument('--print_every', type=int, default=50, help='Print interval')
    train_parser.add_argument('--eta', type=float, default=0.02, help='Learning rate')
    train_parser.add_argument('--plot', action='store_true', help='Whether to plot results')
    train_parser.add_argument('--output_file', required=True, help='File to save the trained model')

    # Predict parser
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('--model_file', required=True, help='File containing saved model')
    predict_parser.add_argument('--x0_file', required=True, help='Pickle file containing x0')
    predict_parser.add_argument('--t_file', required=True, help='Pickle file containing t')
    predict_parser.add_argument('--output_file', required=True, help='File to save prediction results')

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
