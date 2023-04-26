####Delete all flags before declare#####
# batch_size = 16

# for name in list(flags.FLAGS):
#       delattr(flags.FLAGS,name)


tf.app.flags.DEFINE_string('f', '', 'kernel')
###### Settings ######
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('training_epoch', 30, 'Number of epochs to train.')
flags.DEFINE_integer('batch_size', 32, 'batch size.')
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
num_nodes = 135
# batch_size = 16

labels = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# ADJ = tf.placeholder(shape=[135, 135], dtype=tf.float64)

inputs1 = tf.placeholder(shape=[None, 36,1], dtype=tf.float32)
inputs2 = tf.placeholder(shape=[None, 10], dtype=tf.float32)
inputs  = [inputs1, inputs2]


weights = weight_variable_glorot(66,135, name='weights')
weights1 = weight_variable_glorot(135,135, name='weights1')


# weights = {
#     'out': tf.Variable(tf.random_normal([64, 1], mean=1.0),dtype=tf.float32, name='weight_o')}
# biases = {
#     'out': tf.Variable(tf.random_normal([1]),dtype=tf.float32,name='bias_o')}

##First hour (define i as = 5) 
i = 5 
# hr = tf.convert_to_tensor(10, dtype=tf.int32)
# dftrain1 = train_ST_emb_reshaped[0:batch_size]
# dftrain2 = train_Ex_emb[0:batch_size]

# train_ST_emb_reshaped,train_Ex_emb
# pred, w1,b1, w2,b2 = Build_Model(df1, 
#                Adj_distance[i],Adj_Speed[i],Adj_Flow[i], weights_ST, bias_ST, weights_ST1, bias_ST1,weights ,biases)

pred,w1,w2 = Build_Model(inputs1,inputs2,Adj_distance[i],Adj_Speed[i],Adj_Flow[i],weights,weights1)
y_pred = pred

##loss
mae_loss = tf.keras.losses.MeanAbsoluteError()
loss =  tf.reduce_mean(mae_loss(y_pred,labels))
# loss = tf.reduce_mean(tf.keras.metrics.mean_absolute_error(y_pred,labels))
# loss = tf.reduce_mean(tf.square(y_pred-labels))
# loss = tf.reduce_mean(tf.squared_difference(y_pred, labels))
##rmse

# error = tf.reduce_mean(tf.abs(labels - y_pred))
error = tf.reduce_mean(tf.keras.metrics.mean_absolute_error(labels, y_pred))
# error = tf.reduce_mean(tf.square(y_pred-labels))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)


###### Initialize session ######
# variables = tf.initialize_all_variables()
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())  
#sess = tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())


def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# path = './model'
    
###### evaluation ######
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae  = mean_absolute_error(a,b)
    mape = MAPE (a,b)
#     F_norm = la.norm(a-b,'fro')/la.norm(a,'fro')
#     r2  = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
#     var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae, mape

H_RMSE, H_MAE, H_MAPE = [], [], []
model_rmse, model_mae, model_mape = [], [], []
x_axe,batch_loss,batch_rmse,batch_pred = [], [], [], []
test_loss,test_rmse,test_mae,test_pred, test_map = [],[],[],[],[]
# time_start = time.time()
print(f'We start train model')
for i in range(5,23):
    train_ST_emb_reshaped, train_Ex_emb, df_trainy   = pre_process(df_Train, i)
    test_ST_emb_reshaped, test_Ex_emb, df_testy       = pre_process(df_Test, i)
    totalbatch = int(train_ST_emb_reshaped.shape[0]/batch_size)
    training_data_count = len(train_ST_emb_reshaped)
    totalbatchT = int(test_ST_emb_reshaped.shape[0]/batch_size)
    MAE1  = np.empty([totalbatchT], dtype=float)
    MAE1.fill(1000)
    RMSE1 = np.empty([totalbatchT], dtype=float)
    RMSE1.fill(1000)
    MAPE1  = np.empty([totalbatchT], dtype=float)
    MAPE1.fill(1000)
    
    print(f'We train model for hour {i}')
#     print(f'train_size= ({train_ST_emb_reshaped.shape[0]},{train_Ex_emb.shape[0]}, {df_trainy.shape[0]})')
#     print(f'test_size= ({test_ST_emb_reshaped.shape[0]},{test_Ex_emb.shape[0]}, {df_testy.shape[0]})')
    if i ==5:
        pred,w1,w2 = Build_Model(inputs1,inputs2,Adj_distance[i-5],Adj_Speed[i-5],Adj_Flow[i-5],weights,weights1)
    else:
        pred,w1,w2 = Build_Model(inputs1,inputs2,Adj_distance[i-5],Adj_Speed[i-5],Adj_Flow[i-5],w1,w2)
    for epoch in range(training_epoch):
        print(f'we start train epoch:{epoch}')
#         st_epoch = time.time()
        for m in range(totalbatch):
            mini_batch1 = train_ST_emb_reshaped[m * batch_size : (m+1) * batch_size]
            mini_label  = df_trainy[m * batch_size : (m+1) * batch_size]
            mini_batch2 = train_Ex_emb[m * batch_size : (m+1) * batch_size]
            op, loss1, rmse1, train_output = sess.run([optimizer, loss, error, y_pred],
                                                     feed_dict = {inputs[0]:mini_batch1, inputs[1]:mini_batch2, labels:mini_label})
            batch_loss.append(loss1)
#             max_value = np.max(mini_label) 
            batch_rmse.append(rmse1)
#             et_epoch = time.time()
#             epoch_time = et_epoch - st_epoch
#         print(f'batch RMSE:{rmse1}')
#         print(f'Train_batch Loss:{loss1}')
#         op.get_weights()
#         print(f'batch ACC:{acc}')
#     print(f'w is:',WW[0][0][0:5])
#         print(f"Epoch time {epoch} is: {epoch_time} sec")

        test_lossb, test_predb,test_rmseb,test_maeb = [],[],[],[]
    
        for m in range(totalbatchT):
            mini_batchT1 = test_ST_emb_reshaped[m * batch_size : (m+1) * batch_size]
            mini_batchT2 = test_Ex_emb[m * batch_size : (m+1) * batch_size]
            mini_labelT = df_testy[m * batch_size : (m+1) * batch_size]
#         test_T.append(mini_batchT)
#         testy_T.append(mini_labelT)
#     test_T = np.array(test_T)
#     testy_T = np.array(testy_T)
            lossb, rmseb, test_outputb = sess.run([loss, error, y_pred],
                                         feed_dict = {inputs[0]:mini_batchT1, inputs[1]:mini_batchT2,labels:mini_labelT})
            test_labelb = mini_labelT.values
#             print(f'Test_batch Loss:{lossb}')
            rmseb, maeb,mape  = evaluation(test_labelb, test_outputb)
            if maeb < MAE1[m]:
                MAE1[m] = maeb
            if rmseb < RMSE1[m]:
                RMSE1[m] = rmseb
            if mape < MAPE1[m]:
                MAPE1[m] = mape
          
      
    H_RMSE.append(RMSE1)
    H_MAE.append(MAE1)
    H_MAPE.append(MAPE1)
#         rmse = np.mean(np.array(test_rmseb))  
#         mae = np.mean(np.array(test_maeb)) 
#         acc = np.mean(np.array(test_accb)) 
#         r2_score = np.mean(np.array(test_r2b)) 
#         var_score = np.mean(np.array(test_varb)) 
#     test_predb = np.array(test_predb).ravel()
#     test_lossb = np.array(test_lossb).ravel()
#     test_label = df_newT[Target].values
#     max_value1 = np.max(df_newT[Target]) 
#     test_label1 = test_label 
#     test_output1 = test_predb 
#     test_loss.append(test_lossb)
#         test_rmse.append(rmse )
#         test_mae.append(mae)
#         test_acc.append(acc)
#         test_r2.append(r2_score)
#         test_var.append(var_score)
#     test_pred.append(test_output1)
    R,M,P =[],[],[]
    for i in range(len(H_RMSE)):
        R.append(np.mean(np.array(H_RMSE[i])))
        M.append(np.mean(np.array(H_MAE[i])))
        P.append(np.mean(np.array(H_MAPE[i])))

    model_rmse.append(np.mean(np.array(R)))
    model_mae.append(np.mean(np.array(M)))
    model_mape.append(np.mean(np.array(P)))
    print(f'Hour {i+5} RMSE:{np.mean(np.array(R))}')
    print(f'Hour {i+5} MAE:{np.mean(np.array(M))}')
    print(f'Hour {i+5} MAPE:{np.mean(np.array(P))}')
            
#     h_RMSE = np.min(test_rmse)
#     h_MAE  = np.min(test_mae)
#     print(f'RMSE for Hour is : {h_RMSE}')
#     print(f'MAE for Hour is : {h_MAE}')

        
#     H_RMSE.append(h_RMSE)
#     H_MAE.append(h_MAE)
#     h_RMSE = 0
#     h_MAE  = 0
#     h_ACC  = 0
        

        
print("Model RMSE = ", np.mean(np.array(model_rmse)))
print("Model MAE  = ", np.mean(np.array(model_mae)))
print("Model MAPE = ", np.mean(np.array(model_mape)))
