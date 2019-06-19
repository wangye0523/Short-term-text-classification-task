import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import sys
reload(sys)
sys.setdefaultencoding('utf-8')



from utils import build_rnn_data_feed , get_test_sigmas
from turbo_rnn import load_model
import keras.backend as K
import sys
import numpy as np
import time
import keras
import tensorflow as tf

import commpy.channelcoding.convcode as cc
import commpy.channelcoding.interleavers as RandInterlv


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_block_train', type=int, default=50000)
    parser.add_argument('-num_block_test', type=int, default=50000)
    parser.add_argument('-block_len', type=int, default=400)
    parser.add_argument('-num_hidden_unit', type=int, default=500)
    parser.add_argument('-batch_size',  type=int, default=256)
    parser.add_argument('-num_epoch',  type=int, default=100)
    parser.add_argument('-train_snr', type=float, default=5.0)
    parser.add_argument('-snr_test_start', type=float, default=-10)
    parser.add_argument('-snr_test_end', type=float, default=10)
    parser.add_argument('-snr_points', type=int, default=21)

    parser.add_argument('-enc1',  type=int, default=7)
    parser.add_argument('-enc2',  type=int, default=5)
    parser.add_argument('-feedback',  type=int, default=7)
    parser.add_argument('-M',  type=int, default=2, help="Number of delay elements in the convolutional encoder")
    parser.add_argument('-num_dec_iteration', type=int, default=6)


    parser.add_argument('-learning_rate',  type=float, default=0.001)

    parser.add_argument('-noise_type', choices = ['awgn', 't-dist','hyeji_bursty'], default='awgn')
    parser.add_argument('-train_loss', choices = ['binary_crossentropy', 'mse', 'mae'], default='binary_crossentropy')

    parser.add_argument('-radar_power', type=float, default=20.0)
    parser.add_argument('-radar_prob', type=float, default=0.05)

    parser.add_argument('-fixed_var', type=float, default=0.00)
    parser.add_argument('--GPU_proportion', type=float, default=0.90)
    parser.add_argument('-id', type=str, default=str(np.random.random())[2:8])

    args = parser.parse_args()
    print args
    print '[ID]', args.id
    return args

if __name__ == '__main__':
    args = get_args()

    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    frac = args.GPU_proportion
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = frac
    set_session(tf.Session(config=config))

    ##########################################
    # Setting Up Channel & Train SNR
    ##########################################

    M = np.array([args.M])
    generator_matrix = np.array([[args.enc1,args.enc2]])
    feedback = args.feedback
    trellis1 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
    trellis2 = cc.Trellis(M, generator_matrix,feedback=feedback)# Create trellis data structure
    interleaver = RandInterlv.RandInterlv(args.block_len, 0)
    p_array = interleaver.p_array
    print '[Convolutional Code Codec] Encoder', 'M ', M, ' Generator Matrix ', generator_matrix, ' Feedback ', feedback
    codec  = [trellis1, trellis2, interleaver]
    train_snr_Es = args.train_snr + 10*np.log10(float(args.block_len)/float(2*args.block_len))
    sigma_snr  = np.sqrt(1/(2*10**(float(train_snr_Es)/float(10))))
    SNR = -10*np.log10(sigma_snr**2)
    noiser = [args.noise_type, sigma_snr]  # For now only AWGN is supported
    X_feed_test, X_message_test = build_rnn_data_feed(args.num_block_test,  args.block_len, noiser, codec)
    X_feed_train,X_message_train= build_rnn_data_feed(args.num_block_train, args.block_len, noiser, codec)

    from keras.layers import Conv1D,MaxPool1D,Dense,Input,Flatten,Lambda,BatchNormalization,Dropout
    from keras.models import Model
    input_shape_tmp=(X_feed_train.shape[1],X_feed_train.shape[2])

    inputs=Input(shape=input_shape_tmp)
    def split_data_0(x):
        x1 = x[:,:,0:1]
        return x1
    def split_data_1(x):
        x1 = x[:,:,1:2]
        return x1
    def split_data_2(x):
        xx = x[:,:,2:3]
        return xx
    def concat(x):
        return K.concatenate(x)


    Lambda_concat=Lambda(concat)
    sys_r=Lambda(split_data_0)(inputs)
    par_1=Lambda(split_data_1)(inputs)
    par_2=Lambda(split_data_2)(inputs)
    x1 = Conv1D(50,5,activation='relu',padding='same')(par_1)
    x1 = Dropout(rate=0.3)(x1)
    x1 = MaxPool1D(pool_size=8,padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(10,5,activation='relu',padding='same')(x1)
    x1 = Dropout(rate=0.3)(x1)
    x1 = MaxPool1D(pool_size=5,padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Flatten()(x1)

    x2 = Conv1D(50,5,activation='relu',padding='same')(par_2)
    x2 = Dropout(rate=0.3)(x2)
    x2 = MaxPool1D(pool_size=8,padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(10,5,activation='relu',padding='same')(x2)
    x2 = Dropout(rate=0.3)(x2)
    x2 = MaxPool1D(pool_size=5,padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Flatten()(x2)

    x3 = Flatten()(sys_r)
    x = Lambda_concat([x1,x2,x3])
    x = Dense(units=args.num_hidden_unit,activation='relu')(x)

    x = Dense(X_message_train.shape[1],activation='sigmoid')(x)
    model = Model(inputs=inputs,outputs=x)

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    history=model.fit(x=X_feed_train, y=X_message_train, batch_size=args.batch_size,
              epochs=args.num_epoch,verbose=0,validation_split=0.2)  # starts training
    import matplotlib.pyplot as plt
    print max(history.history['val_acc'])
    print min(history.history['loss'])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    SNRS, test_sigmas = get_test_sigmas(args.snr_test_start, args.snr_test_end, args.snr_points)

    turbo_res_ber = []
    turbo_res_bler= []

    for idx in xrange(len(test_sigmas)):
        start_time = time.time()
        noiser = [args.noise_type, test_sigmas[idx]]
        X_feed_test, X_message_test = build_rnn_data_feed(args.num_block_test, args.block_len, noiser, codec)
        pd       = model.predict(X_feed_test,batch_size=args.batch_size)
        decoded_bits = np.round(pd)

        # Compute BER and BLER
        ber_err_rate  = (sum(sum(abs(decoded_bits-X_message_test))))*1.0/(X_message_test.shape[0]*X_message_test.shape[1])# model.evaluate(X_feed_test, X_message_test, batch_size=10)
        tp0 = (abs(decoded_bits-X_message_test)).reshape([X_message_test.shape[0],X_message_test.shape[1]])
        bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_message_test.shape[0])

        print '[testing] This is SNR', SNRS[idx] , 'RNN BER ', ber_err_rate, 'RNN BLER', bler_err_rate
        turbo_res_ber.append(ber_err_rate)
        turbo_res_bler.append(bler_err_rate)
        end_time = time.time()
        print '[testing] runnig time is', str(end_time-start_time)

    print '[Result Summary] SNRS is', SNRS
    print '[Result Summary] Turbo RNN BER is', turbo_res_ber
    print '[Result Summary] Turbo RNN BLER is', turbo_res_bler

    batch=[64,128,256,512,1024,2048]
    turbo_throughput=[]
    for tmp_batch in batch:
        s_1=time.time()
        model.predict(X_feed_test,batch_size=tmp_batch)
        s_2=time.time()
        time_tmp=s_2-s_1
        turbo_throughput.append(args.num_block_test*args.block_len*1.0/time_tmp)

    print '[Result Summary] Turbo RNN Throughput is', turbo_throughput

from sklearn.decomposition import PCA

# def pca(X_feed_train):
#     sys_r, par1_r, par2_r = X_feed_train[:,:,0:1],X_feed_train[:,:,1:2],X_feed_train[:,:,2:3]
#     par1_r=par1_r.reshape((par1_r.shape[0],par1_r.shape[1]))
#     par2_r=par2_r.reshape((par2_r.shape[0],par2_r.shape[1]))
#     sys_r=sys_r.reshape((sys_r.shape[0],sys_r.shape[1]))
#
#     pca=PCA(n_components=270)
#     par1_r_pca = pca.fit_transform(par1_r)
#     par2_r_pca = pca.fit_transform(par2_r)
#     sys_r_pca = pca.fit_transform(sys_r)
#     X_feed_train=np.concatenate((sys_r_pca.reshape(-1,sys_r_pca.shape[1],1),par1_r_pca.reshape(-1,par2_r_pca.shape[1],1),par2_r_pca.reshape(-1,par2_r_pca.shape[1],1)),axis=2)
#     return X_feed_train
#
# X_feed_train=pca(X_feed_train)
# X_feed_test=pca(X_feed_test)
