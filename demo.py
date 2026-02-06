import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from scipy.io import savemat
from sklearn import preprocessing
from sklearn.manifold import TSNE
from torch import optim
from torch.autograd import Variable

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import os

from ConvKAN3D import effConvKAN3D



from MSFE import MultiScaleEnhancement



parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['BayArea', 'Barbara', 'farmland','river','SA'], default='farmland', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=15, help='number of evaluation')
parser.add_argument('--patches', type=int, default=5, help='number of patches')
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
parser.add_argument('--epoches', type=int, default=1, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--train_number', type=int, default=500, help='train_number')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
#-------------------------------------------------------------------------------
# 定位训练和测试样本
def chooose_train_and_test_point(train_data, test_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    # number_true = []
    # pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class
    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)
    return total_pos_train, total_pos_test, number_train, number_test
#-------------------------------------------------------------------------------
# 边界拓展：镜像
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
#-------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image

def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band
#-------------------------------------------------------------------------------
# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, test_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    # x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("**************************************************")

    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    print("**************************************************")
    return x_train_band, x_test_band
#-------------------------------------------------------------------------------
# 标签y_train, y_test
def train_and_test_label(number_train, number_test, num_classes):
    y_train = []
    y_test = []
    # y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("**************************************************")
    return y_train, y_test
#-------------------------------------------------------------------------------
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()
#-------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])  # 用于存储目标标签
    pre = np.array([])  # 用于存储预测结果

    for batch_idx, (batch_data_t1, batch_data_t2, batch_target) in enumerate(train_loader):
        batch_data_t1 = batch_data_t1.cuda()
        batch_data_t2 = batch_data_t2.cuda()
        batch_target = batch_target.cuda()

        optimizer.zero_grad()

        # 模型返回两个值
        batch_pred, loss_DA = model(batch_data_t1, batch_data_t2)

        # 确保 batch_pred 是张量
        if not isinstance(batch_pred, torch.Tensor):
            raise TypeError(f"batch_pred must be a Tensor, but got {type(batch_pred)}")

        # 展平模型输出
        # 假设模型输出是 [batch_size, num_classes, height, width]
        batch_pred = batch_pred.view(batch_pred.size(0), -1)  # [batch_size, num_classes]

        # 确保目标张量是 [batch_size] 的一维张量
        if batch_target.dim() != 1:
            batch_target = batch_target.view(-1)  # 将目标展平为一维

        # 计算分类损失
        loss_cls = criterion(batch_pred, batch_target)

        # 总损失
        loss = loss_cls + loss_DA

        loss.backward()
        optimizer.step()

        # 记录训练指标
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data_t1.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre


 #-------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])  # 用于存储目标标签
    pre = np.array([])  # 用于存储预测结果

    for batch_idx, (batch_data_t1, batch_data_t2, batch_target) in enumerate(valid_loader):
        batch_data_t1 = batch_data_t1.cuda()
        batch_data_t2 = batch_data_t2.cuda()
        batch_target = batch_target.cuda()

        # 解包模型返回值
        batch_pred, loss_DA = model(batch_data_t1, batch_data_t2)

        # 确保 batch_pred 是张量
        if not isinstance(batch_pred, torch.Tensor):
            raise TypeError(f"batch_pred must be a Tensor, but got {type(batch_pred)}")

        # 检查 batch_pred 的形状
        # 如果模型输出是 [batch_size, num_classes, height, width]，则需要调整目标
        if batch_pred.dim() == 4:  # 如果是四维张量，假设为图像数据
            batch_pred = batch_pred.view(batch_pred.size(0), -1)  # 展平为 [batch_size, num_classes]

        # 确保目标张量是 [batch_size] 的一维张量
        if batch_target.dim() != 1:
            batch_target = batch_target.view(-1)  # 展平为一维张量

        # 计算分类损失
        loss_cls = criterion(batch_pred, batch_target)

        # 总损失（如果需要结合领域对齐损失）
        loss = loss_cls + loss_DA

        # 计算精度
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data_t1.shape[0]

        # 更新指标
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre






def test_epoch(model, test_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])  # 用于存储目标标签
    pre = np.array([])  # 用于存储预测结果

    for batch_idx, (batch_data_t1, batch_data_t2, batch_target) in enumerate(test_loader):
        batch_data_t1 = batch_data_t1.cuda()
        batch_data_t2 = batch_data_t2.cuda()
        batch_target = batch_target.cuda()

        # 解包元组
        batch_pred, _ = model(batch_data_t1, batch_data_t2)  # 只取第一个值

        # 确保 batch_pred 是张量
        if not isinstance(batch_pred, torch.Tensor):
            raise TypeError(f"batch_pred must be a Tensor, but got {type(batch_pred)}")

        # 如果模型输出是 [batch_size, num_classes, height, width]，则展平
        if batch_pred.dim() == 4:  # 如果是四维张量
            batch_pred = batch_pred.view(batch_pred.size(0), -1)  # 展平成 [batch_size, num_classes]

        # 计算 topk
        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())

    return pre
#-------------------------------------------------------------------------------
def output_metric(tar, pre):
    # 计算混淆矩阵
    matrix = confusion_matrix(tar, pre)
    # 计算各项指标，包括 P, R, F1
    OA, AA_mean, Kappa, AA, P, R, F1 = cal_results(matrix)
    return OA, AA_mean, Kappa, AA, P, R, F1

#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)

    # 计算 TP, TN, FP, FN
    TP = np.diag(matrix)  # 对角线元素为 TP
    FP = np.sum(matrix, axis=0) - TP  # 每列和减去 TP
    FN = np.sum(matrix, axis=1) - TP  # 每行和减去 TP
    TN = np.sum(matrix) - (FP + FN + TP)  # 总和减去 (FP + FN + TP)

    # 初始化 P, R, F1
    P = np.zeros([shape[0]], dtype=float)
    R = np.zeros([shape[0]], dtype=float)
    F1 = np.zeros([shape[0]], dtype=float)

    esp = 1e-6  # 避免除零

    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / (np.sum(matrix[i, :]) + esp)
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])

        # 计算 Precision, Recall, F1
        P[i] = TP[i] / (TP[i] + FP[i] + esp)
        R[i] = TP[i] / (TP[i] + FN[i] + esp)
        F1[i] = 2 * P[i] * R[i] / (P[i] + R[i] + esp)

    # 总精度 (Overall Accuracy, OA)
    OA = number / np.sum(matrix)

    # 平均精度 (Average Accuracy, AA_mean)
    AA_mean = np.mean(AA)

    # Kappa系数
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)

    # 平均 P, R, F1
    P_mean = np.mean(P)
    R_mean = np.mean(R)
    F1_mean = np.mean(F1)

    return OA, AA_mean, Kappa, AA, P_mean, R_mean, F1_mean
#-------------------------------------------------------------------------------
# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
# prepare data
if args.dataset == 'BayArea':
    data_t1 = loadmat("D:\change detection\Datasets\ChangeDetectionDataset-master/bayArea\mat\Bay_Area_2013.mat")['HypeRvieW']
    data_t2 = loadmat("D:\change detection\Datasets\ChangeDetectionDataset-master/bayArea\mat\Bay_Area_2015.mat")['HypeRvieW']
    data_label = loadmat("D:\change detection\Datasets\ChangeDetectionDataset-master/bayArea\mat/bayArea_gtChanges2.mat.mat")['HypeRvieW']
    uc_position = np.array(np.where(data_label==2)).transpose(1,0)
    c_position = np.array(np.where(data_label==1)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))

    total_samples = c_position.shape[0] + uc_position.shape[0]
    args.train_number = int(total_samples * 0.05)

elif args.dataset == 'Barbara':
    data_t1 = loadmat("D:\change detection\Datasets\ChangeDetectionDataset-master\santaBarbara\mat/barbara_2013.mat")['HypeRvieW']
    data_t2 = loadmat("D:\change detection\Datasets\ChangeDetectionDataset-master\santaBarbara\mat/barbara_2014.mat")['HypeRvieW']
    data_label = loadmat("D:\change detection\Datasets\ChangeDetectionDataset-master\santaBarbara\mat/barbara_gtChanges.mat")['HypeRvieW']
    uc_position = np.array(np.where(data_label==2)).transpose(1,0)
    c_position = np.array(np.where(data_label==1)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))

    total_samples = c_position.shape[0] + uc_position.shape[0]
    args.train_number = int(total_samples * 0.05)

elif args.dataset == 'SA':
    data_t1 = loadmat("D:\change detection\Datasets\Sa\Sa1.mat")['T1']
    data_t2 = loadmat("D:\change detection\Datasets\Sa\Sa2.mat")['T2']
    data_label = loadmat("D:\change detection\Datasets\Sa\SaGT_new.mat")['GT']
    uc_position = np.array(np.where(data_label==2)).transpose(1,0)
    c_position = np.array(np.where(data_label==1)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))

    total_samples = c_position.shape[0] + uc_position.shape[0]
    args.train_number = int(total_samples * 0.05)



elif args.dataset == 'river':
    data_t1 = loadmat("D:\change detection\Datasets\River\River_before.mat")['river_before']
    data_t2 = loadmat('D:\change detection\Datasets\River\River_after.mat')['river_after']
    data_label = loadmat("D:\change detection\Datasets\River\Rivergt.mat")['gt']



    # 修改筛选条件，假设未改变区域的值为1，改变区域的值为2
    uc_position = np.array(np.where(data_label == 1)).transpose(1, 0)
    c_position = np.array(np.where(data_label == 2)).transpose(1, 0)

    print((uc_position.shape[0], c_position.shape[0]))

    # 正常化和标签调整
    data_label = (data_label - data_label.min()) / (data_label.max() - data_label.min())
    data_label[data_label == 0] = 2

    total_samples = c_position.shape[0] + uc_position.shape[0]
    args.train_number = int(total_samples * 0.05)

elif args.dataset == 'farmland':
    data_t1 = loadmat("D:\change detection\DASKAN\DASKAN\data/farmland\China_Change_Dataset.mat")['T1']
    data_t2 = loadmat('D:\change detection\DASKAN\DASKAN\data/farmland\China_Change_Dataset.mat')['T2']
    data_label = loadmat('D:\change detection\DASKAN\DASKAN\data/farmland\China_Change_Dataset.mat')['Binary']
    uc_position = np.array(np.where(data_label==0)).transpose(1,0)
    c_position = np.array(np.where(data_label==1)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))
    data_label[data_label==0]=2

    total_samples = c_position.shape[0] + uc_position.shape[0]
    args.train_number = int(total_samples * 0.01)
else:
    raise ValueError("Unkknow dataset")








H1, W1, B1=data_t1.shape
H2, W2, B2=data_t2.shape
TT1=data_t1.reshape(H1*W1, B1)
TT2=data_t2.reshape(H2*W2, B2)
T1=preprocessing.scale(TT1)
T2=preprocessing.scale(TT2)
Time1=T1.reshape(H1, W1, B1)
Time2=T2.reshape(H2, W2, B2)

selected_uc = np.random.choice(uc_position.shape[0], int(args.train_number), replace = False)
selected_c = np.random.choice(c_position.shape[0], int(args.train_number), replace = False)
selected_uc_position=uc_position[selected_uc]
selected_c_position=c_position[selected_c]
TR = np.zeros(data_label.shape)
for i in range (int(args.train_number)):
    TR[selected_c_position[i][0],selected_c_position[i][1]]=1
    TR[selected_uc_position[i][0],selected_uc_position[i][1]]=2
#--------------测试样本-----------------
TE=data_label-TR

# color_mat = loadmat('./data/AVIRIS_colormap.mat')


num_classes = np.max(TR)
num_classes=int(num_classes)
# color_mat_list = list(color_mat)
# color_matrix = color_mat[color_mat_list[3]] #(17,3)
# normalize data by band norm
input1_normalize = np.zeros(data_t1.shape)
input2_normalize = np.zeros(data_t1.shape)
for i in range(data_t1.shape[2]):
    input_max = max(np.max(data_t1[:,:,i]),np.max(data_t2[:,:,i]))
    input_min = min(np.min(data_t1[:,:,i]),np.min(data_t2[:,:,i]))
    input1_normalize[:,:,i] = (data_t1[:,:,i]-input_min)/(input_max-input_min)
    input2_normalize[:,:,i] = (data_t2[:,:,i]-input_min)/(input_max-input_min)

height, width, band = data_t1.shape
print("height={0},width={1},band={2}".format(height, width, band))
#-------------------------------------------------------------------------------

if args.flag_test=='train':
    total_pos_train, total_pos_test, number_train, number_test = chooose_train_and_test_point(TR, TE, num_classes)
    mirror_image_t1 = mirror_hsi(height, width, band, input1_normalize, patch=args.patches)
    mirror_image_t2 = mirror_hsi(height, width, band, input2_normalize, patch=args.patches)
    x_train_band_t1, x_test_band_t1 = train_and_test_data(mirror_image_t1, band, total_pos_train, total_pos_test, patch=args.patches, band_patch=args.band_patches)
    x_train_band_t2, x_test_band_t2 = train_and_test_data(mirror_image_t2, band, total_pos_train, total_pos_test, patch=args.patches, band_patch=args.band_patches)
    y_train, y_test = train_and_test_label(number_train, number_test, num_classes)
    #-------------------------------------------------------------------------------
    # load data
    x_train_t1=torch.from_numpy(x_train_band_t1.transpose(0,2,1)).type(torch.FloatTensor) #[695, 200, 7, 7]
    x_train_t2=torch.from_numpy(x_train_band_t2.transpose(0,2,1)).type(torch.FloatTensor) #[695, 200, 7, 7]
    y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
    Label_train=Data.TensorDataset(x_train_t1,x_train_t2,y_train)
    x_test_t1=torch.from_numpy(x_test_band_t1.transpose(0,2,1)).type(torch.FloatTensor) # [9671, 200, 7, 7]
    x_test_t2=torch.from_numpy(x_test_band_t2.transpose(0,2,1)).type(torch.FloatTensor) # [9671, 200, 7, 7]
    y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
    Label_test=Data.TensorDataset(x_test_t1,x_test_t2,y_test)


    label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
    label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)

elif args.flag_test=='test':
    mirror_image_t1 = mirror_hsi(height, width, band, input1_normalize, patch=args.patches)
    mirror_image_t2 = mirror_hsi(height, width, band, input2_normalize, patch=args.patches)
    x1_true = np.zeros((height*width, args.patches, args.patches, band), dtype=float)
    x2_true = np.zeros((height*width, args.patches, args.patches, band), dtype=float)
    y_true=[]
    for i in range(height):
        for j in range(width):
            x1_true[i*width+j,:,:,:]=mirror_image_t1[i:(i+args.patches),j:(j+args.patches),:]
            x2_true[i*width+j,:,:,:]=mirror_image_t2[i:(i+args.patches),j:(j+args.patches),:]
            y_true.append(i)
    y_true = np.array(y_true)
    x1_true_band = gain_neighborhood_band(x1_true, band, args.band_patches, args.patches)
    x2_true_band = gain_neighborhood_band(x2_true, band, args.band_patches, args.patches)
    x1_true_band=torch.from_numpy(x1_true_band.transpose(0,2,1)).type(torch.FloatTensor)
    x2_true_band=torch.from_numpy(x2_true_band.transpose(0,2,1)).type(torch.FloatTensor)
    y_true=torch.from_numpy(y_true).type(torch.LongTensor)
    Label_true=Data.TensorDataset(x1_true_band,x2_true_band,y_true)
    label_true_loader=Data.DataLoader(Label_true,batch_size=100,shuffle=False)
    print('------测试数据加载完毕------')
#-------------------------------------------------------------------------------
# create model

from fast_kan import FastKAN
from ConvKAN import ConvKAN
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class QuadrupleSKAN(nn.Module):
    def __init__(self, num_classes=2):
        super(QuadrupleSKAN, self).__init__()

        self.in_chs = 154

        # 定义 KAN2D 层
        self.ConvKAN1 = ConvKAN(in_channels=self.in_chs, out_channels=8, kernel_size=3, padding=1)
        self.ConvKAN2 = ConvKAN(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.ConvKAN3 = ConvKAN(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.ConvKAN4 = ConvKAN(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # 定义 1x1 卷积以匹配通道数（用于残差连接）
        self.res_conv1 = nn.Conv2d(self.in_chs, 8, kernel_size=1)
        self.res_conv2 = nn.Conv2d(8, 16, kernel_size=1)
        self.res_conv3 = nn.Conv2d(16, 32, kernel_size=1)
        self.res_conv4 = nn.Conv2d(32, 64, kernel_size=1)

        # 使用 AdaptiveAvgPool2d 来减少信息丢失
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 定义 FastKAN 层
        self.FastKAN = FastKAN([64, 16, num_classes])

    def forward(self, x):
        # 确保输入是四维张量
        if x.dim() == 3:
            x = x.unsqueeze(-1)  # 将 x 从 [batch_size, channels, height] 调整为 [batch_size, channels, height, 1]

        # 第一个残差连接
        identity = self.res_conv1(x)
        x = self.ConvKAN1(x)
        x += identity  # 残差连接

        # 第二个残差连接
        identity = self.res_conv2(x)
        x = self.ConvKAN2(x)
        x += identity  # 残差连接

        # 第三个残差连接
        identity = self.res_conv3(x)
        x = self.ConvKAN3(x)
        x += identity  # 残差连接

        # 第四个残差连接
        identity = self.res_conv4(x)
        x = self.ConvKAN4(x)
        x += identity  # 残差连接

        # 池化层
        x = self.pool(x)

        # Flatten 特征
        x = torch.flatten(x, 1)

        # 通过 FastKAN 进行处理
        x = self.FastKAN(x)

        return x




class SSTKAN(nn.Module):
    def __init__(self, bands, num_classes):
        super(SSTKAN, self).__init__()
        self.num_classes = num_classes

        # 1. 特征提取（原始模型部分）
        self.QuadrupleSKAN = QuadrupleSKAN(num_classes=num_classes)

        # 2. 用于调整 diff1 和 diff2 的通道数（将原始的 2 维映射到 64 维）
        self.fc_adjust_diff1 = nn.Linear(2, 64)
        self.fc_adjust_diff2 = nn.Linear(2, 64)

        # 3. 3D 卷积模块，将 64 通道映射为 32 通道
        self.effConvKAN3D = effConvKAN3D(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        # 4. 多尺度增强模块，输入通道设置为 32，与 3D 卷积输出匹配
        self.ms_enhance = MultiScaleEnhancement(in_channels=32, reduction_factor=1)

        self.features = {}


    def forward(self, Time1, Time2):
        # (1) 特征提取：得到 Time1_fea 与 Time2_fea（形状：[b, num_classes]，例如 num_classes=2）
        Time1_fea = self.QuadrupleSKAN(Time1)
        Time2_fea = self.QuadrupleSKAN(Time2)
        self.features['p_alignment1'] = Time1_fea.detach()
        self.features['p_alignment2'] = Time2_fea.detach()

        # (2) 计算 diff1 和 diff2
        # diff1 为直接差分
        diff1 = Time1_fea - Time2_fea
        # diff2 为拼接后取前 num_classes 个通道（与原模型一致）
        diff2 = torch.cat((Time1_fea, Time2_fea), dim=1)[:, :self.num_classes]

        # 通过全连接层映射到 64 维
        diff1 = self.fc_adjust_diff1(diff1)
        diff2 = self.fc_adjust_diff2(diff2)

        # 如果 diff1 和 diff2 为二维，则扩展为 [b, 64, 1, 1]，以便后续卷积处理
        if diff1.dim() == 2:
            diff1 = diff1.unsqueeze(-1).unsqueeze(-1)
        if diff2.dim() == 2:
            diff2 = diff2.unsqueeze(-1).unsqueeze(-1)

        # 为适应 3D 卷积，将其扩展一维为 [b, 64, depth, H, W]（这里令 depth=1）
        diff1 = diff1.unsqueeze(2)
        diff2 = diff2.unsqueeze(2)

        # 3D 卷积处理：将 64 通道映射为 32 通道
        diff1 = self.effConvKAN3D(diff1).squeeze(2)  # 结果形状 [b, 32, 1, 1]
        diff2 = self.effConvKAN3D(diff2).squeeze(2)  # 结果形状 [b, 32, 1, 1]

        # (3) 在相加之前进行领域对齐：
        # 此处采用 CORAL 损失衡量 diff1 与 diff2 之间分布的差异，
        # 你也可以设计其它对齐策略（例如对 diff1、diff2 分别做归一化或对齐变换）。
        loss_Mean = (diff1.mean() - diff2.mean()) ** 2 * self.num_classes

        loss_center = self.center_alignment_loss(diff1, diff2)
        loss_CORAL = self.CORAL(diff1, diff2)
        loss_variance = (diff1.var()-diff2.var())**2 * self.num_classes
        

        # alpha 为可学习参数，这里直接初始化
        self.alpha = nn.Parameter(torch.tensor(1.0))
        loss_DA =loss_CORAL+loss_Mean

        # (4) branch1：将经过领域对齐处理（损失已加入训练）的 diff1 和 diff2 相加
        branch1 = diff1 + diff2  # 形状 [b, 32, 1, 1]
        diff3 = Time1_fea + Time2_fea
        diff4 = diff3 +branch1
        self.features['pre_alignment1'] = diff1.detach()
        self.features['pre_alignment2'] = diff2.detach()
        self.features['post_alignment1'] = branch1.detach()

        # (5) branch2：对 branch1 经过多尺度增强得到另一分支
        branch2 = self.ms_enhance(diff4)


        result = branch2
        self.features['post_alignment2'] = result.detach()

        # 返回最终融合结果以及领域对齐损失
        return result, loss_DA

    def CORAL(self, X, Y):
        """
        计算 CORAL 损失：输入 X 和 Y 的形状为 [b, C, H, W]，先展平后计算协方差矩阵差异
        """
        X = X.view(X.size(0), -1)
        Y = Y.view(Y.size(0), -1)
        n, dim = X.shape
        m, _ = Y.shape

        # 计算协方差矩阵
        N1 = torch.mean(X, 0, keepdim=True) - X
        M1 = N1.t() @ N1 / (n - 1)
        N2 = torch.mean(Y, 0, keepdim=True) - Y
        M2 = N2.t() @ N2 / (m - 1)
        # 计算 CORAL 损失
        loss = torch.mul((M1 - M2), (M1 - M2))
        loss_coral = torch.sum(loss) / (4 * dim * dim)
        return loss_coral

    def mmd_loss(self, X, Y, kernel="rbf", bandwidth=1.0):
        """
        计算最大均值差异（MMD）损失
        X 和 Y 是 [b, C, H, W] 形状的特征张量
        kernel：指定核函数，默认为RBF核
        bandwidth：核的带宽
        """
        X = X.view(X.size(0), -1)  # 展平
        Y = Y.view(Y.size(0), -1)

        if kernel == "rbf":
            # 计算 RBF 核
            xx = torch.mm(X, X.t())
            yy = torch.mm(Y, Y.t())
            xy = torch.mm(X, Y.t())

            xx_diag = xx.diag().view(-1, 1)
            yy_diag = yy.diag().view(-1, 1)
            dist_X = xx_diag - 2 * xx + xx_diag.t()
            dist_Y = yy_diag - 2 * yy + yy_diag.t()
            dist_X = torch.exp(-dist_X / (2 * bandwidth ** 2))
            dist_Y = torch.exp(-dist_Y / (2 * bandwidth ** 2))
            dist_XY = torch.exp(-2 * xy / (2 * bandwidth ** 2))

            mmd = dist_X.mean() + dist_Y.mean() - 2 * dist_XY.mean()
            return mmd

    def center_alignment_loss(self, X, Y):
        """
        计算中心对齐（Center Alignment）损失：
        目标是最小化 X 和 Y 的均值差异
        X 和 Y 的形状为 [b, C, H, W]，我们计算每个维度的均值差异
        """
        # 计算 X 和 Y 的均值
        mean_X = torch.mean(X, dim=(0, 2, 3), keepdim=True)
        mean_Y = torch.mean(Y, dim=(0, 2, 3), keepdim=True)

        # 计算均值差异并返回损失
        loss = torch.sum((mean_X - mean_Y) ** 2)  # 计算每个通道的均值差异平方
        return loss

    def variance_alignment_loss(self, X, Y):
        """
        计算方差对齐（Variance Alignment）损失：
        目标是最小化 X 和 Y 的方差差异
        X 和 Y 的形状为 [b, C, H, W]，我们计算每个维度的方差差异
        """
        # 计算 X 和 Y 的方差
        var_X = torch.var(X, dim=(0, 2, 3), unbiased=False)  # [C]
        var_Y = torch.var(Y, dim=(0, 2, 3), unbiased=False)  # [C]

        # 计算每个通道的方差差异并返回损失
        loss = torch.sum((var_X - var_Y) ** 2)  # 计算每个通道的方差差异平方
        return loss














model = SSTKAN(bands=B1,num_classes=2)
model = model.cuda()
# criterion
criterion = nn.CrossEntropyLoss().cuda()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=args.gamma)
#-------------------------------------------------------------------------------
# %% 添加t-SNE可视化功能 ----------------------------------------------------------
class FeatureCollector:
    def __init__(self):
        self.diff1_features = []
        self.diff2_features = []
        self.post_features = []
        self.labels = []
        self.expected_dim = None  # 用于记录期望的维度

    def __call__(self, module, input, output):
        # 收集特征前先展平或调整维度
        if 'pre_alignment1' in module.features:
            # 获取 pre_alignment1 特征并展平
            feat1 = module.features['pre_alignment1'].cpu().numpy()
            if feat1.ndim > 2:  # 如果维度大于2，展平后两维
                feat1 = feat1.reshape(feat1.shape[0], -1)
            self.diff1_features.append(feat1)

        if 'pre_alignment2' in module.features:
            # 获取 pre_alignment2 特征并展平
            feat2 = module.features['pre_alignment2'].cpu().numpy()
            if feat2.ndim > 2:  # 如果维度大于2，展平后两维
                feat2 = feat2.reshape(feat2.shape[0], -1)
            self.diff2_features.append(feat2)

        # 收集 post_alignment 特征
        if 'post_alignment1' in module.features:
            feat_post1 = module.features['post_alignment1'].cpu().numpy()
            if feat_post1.ndim > 2:
                feat_post1 = feat_post1.reshape(feat_post1.shape[0], -1)
            self.post_features.append(feat_post1)

    def collect(self, model, dataloader, num_samples=1000):
        model.eval()
        hook = model.register_forward_hook(self)

        with torch.no_grad():
            for i, (t1, t2, labels) in enumerate(dataloader):
                if i * dataloader.batch_size >= num_samples:
                    break
                model(t1.cuda(), t2.cuda())
                # 收集标签（需要确保标签数量与特征数量匹配）
                batch_labels = labels.cpu().numpy()
                # 如果是多分类问题，确保标签正确
                self.labels.extend(batch_labels)

        hook.remove()

        # 确保所有特征列表不为空
        if len(self.diff1_features) == 0 or len(self.diff2_features) == 0 or len(self.post_features) == 0:
            raise ValueError("No features collected. Check if the features are being stored in model.features.")

        # 将所有特征数组连接起来
        diff1_concat = np.concatenate(self.diff1_features, axis=0)
        diff2_concat = np.concatenate(self.diff2_features, axis=0)
        post_concat = np.concatenate(self.post_features, axis=0)

        # 检查维度是否匹配
        print(f"diff1_features shape: {diff1_concat.shape}")
        print(f"diff2_features shape: {diff2_concat.shape}")
        print(f"post_features shape: {post_concat.shape}")

        # 截取相同数量的样本
        min_samples = min(len(diff1_concat), len(diff2_concat), len(post_concat), len(self.labels))
        diff1_concat = diff1_concat[:min_samples]
        diff2_concat = diff2_concat[:min_samples]
        post_concat = post_concat[:min_samples]
        labels_array = np.array(self.labels)[:min_samples]

        return diff1_concat, diff2_concat, post_concat, labels_array


def plot_tsne_3d(features_diff1, features_diff2, features_post, labels, save_path=None):
    """
    美观的配色方案：三张图分开显示
    """
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib as mpl

    # 固定视角
    ELEVATION = 20
    AZIMUTH = 120

    tsne_3d = TSNE(n_components=3, perplexity=30, n_iter=5000, random_state=42)
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 计算并归一化
    embeddings_diff1_norm = scaler.fit_transform(
        tsne_3d.fit_transform(features_diff1.reshape(features_diff1.shape[0], -1))
    )
    embeddings_diff2_norm = scaler.fit_transform(
        tsne_3d.fit_transform(features_diff2.reshape(features_diff2.shape[0], -1))
    )
    embeddings_post_norm = scaler.fit_transform(
        tsne_3d.fit_transform(features_post.reshape(features_post.shape[0], -1))
    )

    unique_labels = np.unique(labels)

    # 定义坐标轴名称
    AXIS_NAMES = ['Spectral Dimension', 'Spatial Dimension', 'Temporal Dimension']

    # 美观的颜色方案选择
    color_schemes = {
        'scheme1': {
            'unchanged': '#2E86AB',  # 深蓝色
            'changed': '#A23B72'  # 深紫色
        },
        'scheme2': {
            'unchanged': '#1B9AAA',  # 青蓝色
            'changed': '#EF476F'  # 珊瑚红
        },
        'scheme3': {
            'unchanged': '#118AB2',  # 海洋蓝
            'changed': '#FF9F1C'  # 落日橙
        },
        'scheme4': {
            'unchanged': '#118AB2',
            'changed': '#EF476F'
        }
    }

    # 选择一种配色方案
    colors = color_schemes['scheme4']  # 可以更换为scheme1, scheme2, scheme3, scheme4

    # 创建三个独立的图形
    figure_names = ['Difference Features', 'Temporal Features', 'Fused Features']
    embeddings_list = [embeddings_diff1_norm, embeddings_diff2_norm, embeddings_post_norm]

    for idx, (embeddings, fig_name) in enumerate(zip(embeddings_list, figure_names)):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 设置整体图形样式
        mpl.rcParams['axes.linewidth'] = 1.5
        mpl.rcParams['grid.linewidth'] = 0.5
        mpl.rcParams['grid.alpha'] = 0.3

        # 绘制未变化类（更精致的样式）
        mask_unchanged = labels == 0
        ax.scatter(embeddings[mask_unchanged, 0],
                   embeddings[mask_unchanged, 1],
                   embeddings[mask_unchanged, 2],
                   color=colors['unchanged'],
                   alpha=0.75,  # 稍微提高透明度
                   s=45,  # 稍微增大点大小
                   label='Unchanged Pixels',
                   edgecolor='white',  # 白色边框
                   linewidth=0.8,  # 边框宽度
                   depthshade=True)  # 深度阴影

        # 绘制变化类
        mask_changed = labels == 1
        ax.scatter(embeddings[mask_changed, 0],
                   embeddings[mask_changed, 1],
                   embeddings[mask_changed, 2],
                   color=colors['changed'],
                   alpha=0.75,
                   s=45,
                   label='Changed Pixels',
                   edgecolor='white',
                   linewidth=0.8,
                   depthshade=True)

        # 设置固定视角
        ax.view_init(elev=ELEVATION, azim=AZIMUTH)

        # 设置坐标轴标签（更美观的字体）
        ax.set_xlabel(AXIS_NAMES[0], fontsize=12, labelpad=12, fontweight='medium')
        ax.set_ylabel(AXIS_NAMES[1], fontsize=12, labelpad=12, fontweight='medium')
        ax.set_zlabel(AXIS_NAMES[2], fontsize=12, labelpad=12, fontweight='medium')

        # 设置坐标轴范围
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        # 设置刻度样式
        ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)
        ax.zaxis.set_tick_params(labelsize=10)

        # 添加网格（更精细）
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

        # 美化坐标平面
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#E0E0E0')
        ax.yaxis.pane.set_edgecolor('#E0E0E0')
        ax.zaxis.pane.set_edgecolor('#E0E0E0')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)

        # 设置背景色
        ax.set_facecolor('#FAFAFA')
        fig.patch.set_facecolor('#FFFFFF')

        # 添加图例（更美观）
        legend = ax.legend(fontsize=11, loc='upper right',
                           frameon=True, fancybox=True,
                           framealpha=0.9, edgecolor='gray')
        legend.get_frame().set_linewidth(0.5)

        # 调整布局
        plt.tight_layout()

        # 保存
        if save_path:
            filename = f'{fig_name.replace(" ", "_")}_Beautiful.png'
            fig.savefig(os.path.join(save_path, filename),
                        bbox_inches='tight', dpi=300,
                        facecolor='white', edgecolor='none',
                        transparent=False)

        plt.show()
if args.flag_test == 'test':
    if args.mode == 'ViT':
        model.load_state_dict(torch.load('./log/ssvit_v2_farmland.pth'))
    elif (args.mode == 'CAF') & (args.patches == 1):
        model.load_state_dict(torch.load('./SpectralFormer_pixel.pt'))
    elif (args.mode == 'CAF') & (args.patches == 7):
        model.load_state_dict(torch.load('./SpectralFormer_patch.pt'))
    else:
        raise ValueError("Wrong Parameters")
    model.eval()
    # output classification maps
    pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
    prediction_matrix = np.zeros((height, width), dtype=float)
    for i in range(height):
        for j in range(width):
            prediction_matrix[i,j] = pre_u[i*width+j] + 1
    plt.subplot(1,1,1)
    color_matrix = np.array([
        [0, 0, 1],  # 蓝色 (RGB: 0, 0, 255)
        [1, 0.5, 0]  # 橙色 (RGB: 255, 127, 0)
    ])


    plt.imshow(prediction_matrix, colors.ListedColormap(color_matrix))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    from PIL import Image
    im=Image.fromarray((1-(prediction_matrix-1))*255)
    import ipdb; ipdb.set_trace()
    savemat('matrix.mat',{'P':prediction_matrix, 'label':label})
elif args.flag_test == 'train':
    print("start training")
    tic = time.time()
    collector = FeatureCollector()
    for epoch in range(args.epoches):
        scheduler.step()

        # 训练模型
        model.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
        OA1, AA_mean1, Kappa1, AA1, P1, R1, F11 = output_metric(tar_t, pre_t)
        print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f} | P: {:.4f} | R: {:.4f} | F1: {:.4f}"
              .format(epoch + 1, train_obj, train_acc, P1, R1, F11))

        output_folder = "D:\change detection\Result"
        os.makedirs(output_folder, exist_ok=True)

        if (epoch % args.test_freq == 0) or (epoch == args.epoches - 1):
            model.eval()
            tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
            OA2, AA_mean2, Kappa2, AA2, P2, R2, F12 = output_metric(tar_v, pre_v)
            print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f} | P: {:.4f} | R: {:.4f} | F1: {:.4f}"
                  .format(OA2, AA_mean2, Kappa2, P2, R2, F12))

            diff1_feats, diff2_feats, post_feats, labels = collector.collect(model, label_test_loader)
            plot_tsne_3d(diff1_feats, diff2_feats, post_feats, labels, save_path=output_folder)


    toc = time.time()
    print("Running Time: {:.2f}".format(toc - tic))
    print("**************************************************")

print("Final result:")
print(
    "OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f} | P: {:.4f} | R: {:.4f} | F1: {:.4f}".format(OA2, AA_mean2, Kappa2, P2, R2,
                                                                                          F12))
print(AA2)
print("**************************************************")

torch.save(model.state_dict(), "log/ssvit_v2_farmland.pth")

x1_true = np.zeros((height*width, args.patches, args.patches, band), dtype=float)
x2_true = np.zeros((height*width, args.patches, args.patches, band), dtype=float)
y_true=[]
for i in range(height):
    for j in range(width):
        x1_true[i*width+j,:,:,:]=mirror_image_t1[i:(i+args.patches),j:(j+args.patches),:]
        x2_true[i*width+j,:,:,:]=mirror_image_t2[i:(i+args.patches),j:(j+args.patches),:]
        y_true.append(i)
y_true = np.array(y_true)
x1_true_band = gain_neighborhood_band(x1_true, band, args.band_patches, args.patches)
x2_true_band = gain_neighborhood_band(x2_true, band, args.band_patches, args.patches)
x1_true_band=torch.from_numpy(x1_true_band.transpose(0,2,1)).type(torch.FloatTensor)
x2_true_band=torch.from_numpy(x2_true_band.transpose(0,2,1)).type(torch.FloatTensor)
y_true=torch.from_numpy(y_true).type(torch.LongTensor)
Label_true=Data.TensorDataset(x1_true_band,x2_true_band,y_true)
label_true_loader=Data.DataLoader(Label_true,batch_size=100,shuffle=False)
criterion = nn.CrossEntropyLoss().cuda()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//20, gamma=args.gamma)
pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
# 你原来把它重塑成矩阵
prediction_matrix = np.zeros((height, width), dtype=float)
for i in range(height):
    for j in range(width):
        prediction_matrix[i, j] = pre_u[i*width + j]  # 注意，这里还没做 +1 或其它操作

# ----------------------------------------------------------
# 如果想让输出矩阵的取值和真值一样：0=背景, 1=未变, 2=变化
# 就需要先弄清你的模型输出含义：
#   - 如果你的网络是2分类(0=未变化, 1=变化)，没有“背景”这个类
#   - 背景可用真值里的0来过滤
# ----------------------------------------------------------

# 假设 data_label 里：0 表示背景，1 表示变化，2 表示未变化（或反过来也行，需你自己确认）
# 而你的网络推理 pre_u：0=未变化, 1=变化
# 则可以写一个简单的映射逻辑：

final_prediction = np.zeros((height, width), dtype=int)

for i in range(height):
    for j in range(width):
        if data_label[i,j] == 0:
            # 真值是背景
            final_prediction[i,j] = 0
        else:
            # 如果不是背景，则直接用网络输出
            #    pre_u=0  => "未变化" => 给它标成 1
            #    pre_u=1  => "变化"   => 给它标成 2
            # 也可以根据你的实际需求，调整具体映射规则
            if prediction_matrix[i,j] == 0:
                final_prediction[i,j] = 1
            else:
                final_prediction[i,j] = 2


plt.figure()
# 这里可以用一个简单灰度或者自定义三色调色板
# 如果你希望 0=黑, 1=灰, 2=白，gray也能直接显示
plt.imshow(final_prediction, cmap='gray')

plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

#output_folder = "D:\change detection\Result"

#mat_file_path = os.path.join(output_folder, "Ours_BayArea.mat")
#savemat(mat_file_path, {"prediction_matrix": final_prediction})

#output_path = os.path.join(output_folder, "Ours_BayArea.png")
#plt.savefig(output_path, bbox_inches='tight', pad_inches=0,  transparent=True)  # 保存图像

plt.show()
#print(f"图像已保存到: {output_path}")


def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))

print_args(vars(args))