import os
import sys
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import numpy as np
import tensorflow as tf
from tqdm import tqdm

### import tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tflib.utils import load_model_from_checkpoint, check_folder, visualize_gt, visualize_progress, save_files
from tflib.nist import load_nist_images
from tflib.qmnist import load_qmnist_attacker_evaluation_set
import lpips_tf
from tflib.gan import Generator


#sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../gan_models/wgangp'))
#from train import *

### Hyperparameters
LAMBDA2 = 0.2
LAMBDA3 = 0.001
RANDOM_SEED = 1000


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-name', type=str, required=True,
                        help='the name of the current experiment (used to set up the save_dir)')
    parser.add_argument('--gan_model_dir', '-gdir', type=str, required=True,
                        help='directory for the Victim GAN model')
    parser.add_argument('--datapath', '-data', type=str,
                        help='the directory for the data')
    parser.add_argument('--data_num', '-dnum', type=int, default=1000,
                        help='the number of query images to be considered')
    parser.add_argument('--batch_size', '-bs', type=int, default=200,
                        help='batch size')
    parser.add_argument('--initialize_type', '-init', type=str, default='zero',
                        choices=['zero',  # 'zero': initialize the z to be zeros
                                 'random',  # 'random': use normal distributed initialization
                                 'nn',  # 'nn' : use nearest-neighbor initialization
                                 ],
                        help='the initialization techniques')
    parser.add_argument('--nn_dir', '-ndir', type=str,
                        help='directory for the fbb(KNN) results')
    parser.add_argument('--results_dir', type=str, default='',
                        help='directory to save the attack results')
    parser.add_argument('--distance', '-dist', type=str, default='l2', choices=['l2', 'l2-lpips'],
                        help='the objective function type')
    parser.add_argument('--if_norm_reg', '-reg', action='store_true', default=True,
                        help='enable the norm regularizer')
    parser.add_argument('--same_census', '-sc', action='store_true', default=False,
                        help='take test data from same census as training (high school) or different. Only when dataset is NIST')
    parser.add_argument('--maxfunc', '-mf', type=int, default=1000,
                        help='the maximum number of function calls')
    parser.add_argument('--dataset', choices=['qmnist', 'mnist', 'nist', 'emnist'], 
                        help='Dataset used to train the model')
    return parser.parse_args()


def check_args(args):
    '''
    check and store the arguments as well as set up the save_dir
    :param args: arguments
    :return:
    '''
    ## load dir
    assert os.path.exists(args.gan_model_dir)

    ## set up save_dir
    if args.results_dir:
        save_dir = os.path.join(args.results_dir, 'wb', args.exp_name)
    else:
        save_dir = os.path.join(os.path.dirname(__file__), 'results/wb', args.exp_name)
    check_folder(save_dir)

    ## store the parameters
    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.writelines(k + ":" + str(v) + "\n")
            print(k + ":" + str(v))
    pickle.dump(vars(args), open(os.path.join(save_dir, 'params.pkl'), 'wb'), protocol=2)
    return args, save_dir, args.gan_model_dir


#############################################################################################################
# main optimization function
#############################################################################################################
def optimize_z(sess, z, x, x_hat,
               init_val_ph, init_val,
               query_imgs, save_dir,
               opt, vec_loss, vec_loss_dict, batch_size):
    """
    z = argmin_z \lambda_1*|x_hat -x|^2  + \lambda_2 * LPIPS(x_hat,x)+ \lambda_3* L_reg
    where x_hat = G(z)

    :param sess:  session
    :param z:  latent variable
    :param x:  query
    :param x_hat: reconstruction
    :param init_val_ph: placeholder for initialization value
    :param init_val: dict that stores the initialization value
    :param query_imgs: query data
    :param save_dir:  save directory
    :param opt: optimization operator
    :param vec_loss: full loss
    :param vec_loss_dict: dict that stores each term in the objective
    :return:
    """

    ### store results
    all_loss = []
    all_z = []
    all_x_hat = []

    ### get the local variables
    vars = [var for var in tf.compat.v1.global_variables() if
            'latent_z' in var.name]
    for v in vars:
        print(v.name)

    ### callback function
    global step, loss_progress
    loss_progress = []
    step = 0

    def update(x_hat_curr, vec_loss_val):
        '''
        callback function for the lbfgs optimizer
        :param x_hat_curr:
        :param vec_loss_val:
        :return:
        '''
        global step, loss_progress
        loss_progress.append(vec_loss_val)
        step += 1

    ### run the optimization for all query data
    size = len(query_imgs)
    for i in tqdm(range(size // batch_size)):
        save_dir_batch = os.path.join(save_dir, str(i))
        print(save_dir_batch)

        try:
            x_gt = query_imgs[i * batch_size:(i + 1) * batch_size]
            x_gt = x_gt.reshape(x_gt.shape[0], x_gt.shape[1], x_gt.shape[2], x_gt.shape[3])
            
            if os.path.exists(save_dir_batch):
                pass
            else:
                visualize_gt(x_gt, check_folder(save_dir_batch))

                ### initialize z
                if init_val_ph is not None:
                    sess.run(tf.compat.v1.variables_initializer(vars),
                             feed_dict={init_val_ph: init_val[i * batch_size:(i + 1) * batch_size]})
                else:
                    sess.run(tf.compat.v1.variables_initializer(vars))

                ### optimize
                loss_progress = []
                step = 0

                vec_loss_curr, z_curr, x_hat_curr = sess.run([vec_loss, z, x_hat], feed_dict={x: x_gt})
                visualize_progress(x_hat_curr, vec_loss_curr, save_dir_batch, step)  # visualize init
                opt.minimize(sess, feed_dict={x: x_gt}, fetches=[x_hat, vec_loss], loss_callback=update)
                vec_loss_curr, z_curr, x_hat_curr = sess.run([vec_loss, z, x_hat], feed_dict={x: x_gt})
                visualize_progress(x_hat_curr, vec_loss_curr, save_dir_batch, step)  # visualize final

                ### store results
                all_loss.append(vec_loss_curr)
                all_z.append(z_curr)
                all_x_hat.append(x_hat_curr)

                ### save to disk
                for key in vec_loss_dict.keys():
                    # each term in the objective
                    val = sess.run(vec_loss_dict[key], feed_dict={x: x_gt})
                    save_files(os.path.join(save_dir, str(i)), [key], [val])
                save_files(os.path.join(save_dir, str(i)),
                           ['full_loss', 'z', 'xhat', 'loss_progress'],
                           [vec_loss_curr, z_curr, x_hat_curr, np.array(loss_progress)])

        except KeyboardInterrupt:
            print('Stop optimization\n')
            break

    try:
        all_loss = np.concatenate(all_loss)
        all_z = np.concatenate(all_z)
        all_x_hat = np.concatenate(all_x_hat)
    except:
        all_loss = np.array(all_loss)
        all_z = np.array(all_z)
        all_x_hat = np.array(all_x_hat)
    return all_loss, all_z, all_x_hat


#############################################################################################################
# main
#############################################################################################################
def main():
    args, save_dir, load_dir = check_args(parse_arguments())
    config_path = os.path.join(load_dir, 'params.pkl')
    if os.path.exists(config_path):
        config = pickle.load(open(config_path, 'rb'))
        Z_DIM = config['Z_DIM']
    else:
        INPUT_WIDTH = 28
        INPUT_HEIGHT = 28
        Z_DIM = 128

    ### open session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:

        ### define variables
        BATCH_SIZE = args.batch_size
        x = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=(None, INPUT_WIDTH, INPUT_HEIGHT, 1), name='x')

        ### initialization
        init_val_ph = None
        init_val = {'pos': None, 'neg': None}
        if args.initialize_type == 'zero':
            z = tf.compat.v1.Variable(tf.compat.v1.zeros([BATCH_SIZE, Z_DIM], tf.compat.v1.float32), name='latent_z')
        elif args.initialize_type == 'random':
            np.random.seed(RANDOM_SEED)
            init_val_np = np.random.normal(size=(Z_DIM,))
            init = np.tile(init_val_np, (BATCH_SIZE, 1)).astype(np.float32)
            z = tf.compat.v1.Variable(init, name='latent_z')
        elif args.initialize_type == 'nn':
            init_val['pos'] = np.load(os.path.join(args.nn_dir, 'pos_z.npy'))[:, 0, :]
            init_val['neg'] = np.load(os.path.join(args.nn_dir, 'neg_z.npy'))[:, 0, :]
            init_val_ph = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, name='init_ph', shape=(BATCH_SIZE, Z_DIM))
            z = tf.compat.v1.Variable(init_val_ph, name='latent_z')
        else:
            raise NotImplementedError

        ### get the reconstruction (x_hat)
        x_hat = Generator(BATCH_SIZE, noise=z)
        x_hat = tf.compat.v1.reshape(x_hat, [-1, 1, INPUT_WIDTH, INPUT_HEIGHT])
        x_hat = tf.compat.v1.transpose(x_hat, perm=[0, 2, 3, 1])

        ### load model
        vars = [v for v in tf.compat.v1.global_variables() if 'latent_z' not in v.name]
        saver = tf.compat.v1.train.Saver(vars)
        sess.run(tf.compat.v1.variables_initializer(vars))
        if_load = load_model_from_checkpoint(load_dir, saver, sess)
        assert if_load is True

        ### loss
        if args.distance == 'l2':
            print('use distance: l2')
            loss_l2 = tf.compat.v1.reduce_mean(tf.compat.v1.square(x_hat - x), axis=[1, 2, 3])
            vec_loss = loss_l2
            vec_losses = {'l2': loss_l2}
        elif args.distance == 'l2-lpips':
            print('use distance: lpips + l2')
            loss_l2 = tf.compat.v1.reduce_mean(tf.compat.v1.square(x_hat - x), axis=[1, 2, 3])
            loss_lpips = lpips_tf.lpips(x_hat, x, normalize=False, model='net-lin', net='vgg', version='0.1')
            vec_losses = {'l2': loss_l2,
                          'lpips': loss_lpips}
            vec_loss = loss_l2 + LAMBDA2 * loss_lpips
        else:
            raise NotImplementedError

        ## regularizer
        norm = tf.compat.v1.reduce_sum(tf.compat.v1.square(z), axis=1)
        norm_penalty = (norm - Z_DIM) ** 2

        if args.if_norm_reg:
            loss = tf.compat.v1.reduce_mean(vec_loss) + LAMBDA3 * tf.compat.v1.reduce_mean(norm_penalty)
            vec_losses['norm'] = norm_penalty
        else:
            loss = tf.compat.v1.reduce_mean(vec_loss)

        ### set up optimizer
        opt = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                     var_list=[z],
                                                     method='L-BFGS-B',
                                                     options={'maxfun': args.maxfunc})

        ### load query images
        if args.dataset=='nist':
            if args.same_census:
                with open(os.path.join(args.datapath, 'HSF_4_images.npy'),'rb') as f:
                    pos_query_imgs = load_nist_images(np.load(f), args.data_num)

                with open(os.path.join(args.datapath, 'HSF_4_images.npy'),'rb') as f:
                    neg_query_imgs = load_nist_images(np.load(f))[30000:30000+args.data_num]
            else:
                with open(os.path.join(args.datapath, 'HSF_4_images.npy'),'rb') as f:
                    pos_query_imgs = load_nist_images(np.load(f), args.data_num)

                with open(os.path.join(args.datapath, 'HSF_6_images.npy'),'rb') as f:
                    neg_query_imgs = load_nist_images(np.load(f), args.data_num)
        elif args.dataset=='qmnist':
            pos_query_imgs, neg_query_imgs, _, _ = load_qmnist_attacker_evaluation_set(args.datapath, args.data_num, args.data_num)


        ### run the optimization on query images
        query_loss, query_z, query_xhat = optimize_z(sess, z, x, x_hat,
                                                     init_val_ph, init_val['pos'],
                                                     pos_query_imgs,
                                                     check_folder(os.path.join(save_dir, 'pos_results')),
                                                     opt, vec_loss, vec_losses, BATCH_SIZE)
        save_files(save_dir, ['pos_loss'], [query_loss])

        query_loss, query_z, query_xhat = optimize_z(sess, z, x, x_hat,
                                                     init_val_ph, init_val['neg'],
                                                     neg_query_imgs,
                                                     check_folder(os.path.join(save_dir, 'neg_results')),
                                                     opt, vec_loss, vec_losses, BATCH_SIZE)
        save_files(save_dir, ['neg_loss'], [query_loss])


if __name__ == '__main__':
    main()