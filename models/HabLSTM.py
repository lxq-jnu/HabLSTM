import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import Module, Sequential, Conv2d

from models.SpatiotemporalLSTM import SpatiotemporalLSTM
from models.TripletAttention import TripletAttention
from utils import print_network

class cx_Diff_MEM(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, forget_bias: float = 0.01):
        r"""
        :param in_channels:              输入通道数
        :param hidden_channels:          隐藏层通道数
        :param kernel_size:              卷积核尺寸
        :param forget_bias:              偏移量
        """
        super(cx_Diff_MEM, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.forget_bias = forget_bias

        padding = (kernel_size // 2, kernel_size // 2)

        kernel_size = (kernel_size, kernel_size)

        self.conv_h_diff = nn.Conv2d(
            in_channels=2 * in_channels, out_channels=hidden_channels * 4,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        self.conv_n = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 3,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        self.conv_w_no = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

    def forward(self, h_diff, n):
        r"""
        :param h_diff:     输入的隐藏层的差分值
        :param n:          状态Tensor
        :return:           n, d
        """
        h_diff_concat = self.conv_h_diff(h_diff)
        h_diff_concat = torch.layer_norm(h_diff_concat, h_diff_concat.shape[1:])
        n_concat = self.conv_n(n)
        n_concat = torch.layer_norm(n_concat, n_concat.shape[1:])

        g_h, i_h, f_h, o_h = torch.split(h_diff_concat, self.hidden_channels, dim=1)
        g_n, i_n, f_n = torch.split(n_concat, self.hidden_channels, dim=1)

        g = torch.tanh(g_h + g_n)
        i = torch.sigmoid(i_h + i_n)
        f = torch.sigmoid(f_h + f_n + self.forget_bias)

        n = f * n + i * g

        o_n = self.conv_w_no(n)
        o_n = torch.layer_norm(o_n, o_n.shape[1:])
        o = torch.sigmoid(o_h + o_n)
        d = o * torch.tanh(n)

        return n, d

class h_Diff_MEM(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, forget_bias: float = 0.01):
        r"""
        :param in_channels:              输入通道数
        :param hidden_channels:          隐藏层通道数
        :param kernel_size:              卷积核尺寸
        :param forget_bias:              偏移量
        """
        super(h_Diff_MEM, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.forget_bias = forget_bias

        padding = (kernel_size // 2, kernel_size // 2)

        kernel_size = (kernel_size, kernel_size)

        self.conv_h_diff = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels * 4,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        self.conv_n = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 3,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        self.conv_w_no = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

    def forward(self, h_diff, n):
        r"""
        :param h_diff:     输入的隐藏层的差分值
        :param n:          状态Tensor
        :return:           n, d
        """
        h_diff_concat = self.conv_h_diff(h_diff)
        h_diff_concat = torch.layer_norm(h_diff_concat, h_diff_concat.shape[1:])
        n_concat = self.conv_n(n)
        n_concat = torch.layer_norm(n_concat, n_concat.shape[1:])

        g_h, i_h, f_h, o_h = torch.split(h_diff_concat, self.hidden_channels, dim=1)
        g_n, i_n, f_n = torch.split(n_concat, self.hidden_channels, dim=1)

        g = torch.tanh(g_h + g_n)
        i = torch.sigmoid(i_h + i_n)
        f = torch.sigmoid(f_h + f_n + self.forget_bias)

        n = f * n + i * g

        o_n = self.conv_w_no(n)
        o_n = torch.layer_norm(o_n, o_n.shape[1:])
        o = torch.sigmoid(o_h + o_n)
        d = o * torch.tanh(n)

        return n, d

class HabLSTM_cell(Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, forget_bias: float = 0.01):
        r"""
        :param in_channels:           输入通道数
        :param hidden_channels:       隐藏层通道数
        :param kernel_size:           卷积核尺寸
        :param forget_bias:           偏移量
        """
        super(HabLSTM_cell, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.forget_bias = forget_bias

        padding = (kernel_size // 2, kernel_size // 2)

        self.conv_x = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels * 7,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )

        self.conv_h = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 4,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )

        self.conv_m = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 3,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )

        # c
        self.conv_i_c = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )
        self.conv_f_c = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )
        self.conv_o_c = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )

        # m
        '''self.conv_i_m = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )
        self.conv_f_m = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )'''
        self.conv_o_m = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )

        self.conv1x1 = nn.Conv2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels,
            kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)
        )

        self.cx_Diff_MEM = cx_Diff_MEM(hidden_channels, hidden_channels, kernel_size)
        self.h_Diff_MEM = h_Diff_MEM(hidden_channels, hidden_channels, kernel_size)

        self.conv_dt = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )

        self.conv_dl = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )

    def forward(self, t_diff, l_diff, x, c, h, m, dt, dl):
        """
        :param h_diff: h_{t}^{l-1} - h_{t-1}^{l-1}
        :param x:    h_{t}^{l-1}
        :param c:    cell记忆信息
        :param h:    时间方向记忆Tensor
        :param m:    空间方向记忆Tensor
        :param n:    MIM-N 记忆Tensor
        :param s:    MIM-S 记忆Tensor
        :return:     更新后的 c, h, m, n, s
        """

        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        m_concat = self.conv_m(m)

        x_concat = torch.layer_norm(x_concat, x_concat.shape[1:])
        h_concat = torch.layer_norm(h_concat, h_concat.shape[1:])
        m_concat = torch.layer_norm(m_concat, m_concat.shape[1:])

        g_x, i_x, f_x, gg_x, ii_x, ff_x, o_x = torch.split(x_concat, self.hidden_channels, dim=1)
        g_h, i_h, o_h , f_h = torch.split(h_concat, self.hidden_channels, dim=1)
        gg_m, ii_m, ff_m = torch.split(m_concat, self.hidden_channels, dim=1)

        g = torch.tanh(g_x + g_h)
        i = torch.sigmoid(i_x + i_h + torch.tanh(self.conv_i_c(c)))

        dt, ft = self.cx_Diff_MEM(t_diff, dt)
        dl, fl = self.h_Diff_MEM(l_diff, dl)

        f = torch.sigmoid(f_x + f_h + torch.tanh(self.conv_f_c(c)) + torch.tanh(self.conv_dt(ft)) + self.forget_bias)

        c = f * c + i * g

        gg = torch.tanh(gg_x + gg_m)
        ii = torch.sigmoid(ii_x + ii_m)
        ff = torch.sigmoid(ff_x + ff_m + self.forget_bias)

        m = ff * m + ii * gg

        o = torch.sigmoid(o_x + o_h + torch.tanh(self.conv_o_c(c)) + torch.tanh(self.conv_o_m(m)) + torch.tanh(self.conv_dl(fl)))

        states = torch.cat([c, m], dim=1)
        #states = states + self.cm_TA(states)

        h = o * torch.tanh(self.conv1x1(states))

        return c, h, m, dt, dl

class HabLSTM(Module):
    def __init__(self, input_size, output_chans, hidden_size=128, filter_size=5, num_layers=4):
        super(HabLSTM, self).__init__()
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        # embedding layer
        self.embed = Conv2d(input_size, hidden_size, 1, 1, 0)

        # lstm layers
        lstm = []
        lstm.append(SpatiotemporalLSTM(in_channels=hidden_size, hidden_channels=hidden_size, kernel_size=filter_size))
        for l in range(1, num_layers):
            lstm.append(HabLSTM_cell(hidden_size, hidden_size, filter_size))
        self.lstm = nn.ModuleList(lstm)

        # output layer
        self.output = Conv2d(hidden_size, output_chans, 1, 1, 0)
        self.TA = TripletAttention()

        '''self.new_Block = new_Block(patch_size=16, in_chans=input_size, embed_dim=768, depth=12,
                                   num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                                   attn_drop_rate=0.1, imgSize=img_size)'''


    def forward(self, x, mask=None, in_len: int = 10, out_len: int = 10):
        device = x[0].device
        batch, _, height, width = x[0].shape

        h = []
        glob_ = []
        glob = []
        c = []
        dt = torch.zeros(batch, self.hidden_size, height, width).to(device)
        dl = torch.zeros(batch, self.hidden_size, height, width).to(device)
        m = torch.zeros(batch, self.hidden_size, height, width).to(device)

        # 初始化最开始的隐藏状态
        for i in range(self.n_layers):
            zero_tensor_h = torch.zeros(batch, self.hidden_size, height, width).to(device)
            zero_tensor_trans = torch.zeros(batch, 2 * self.hidden_size, height, width).to(device)
            c.append(zero_tensor_h)
            h.append(zero_tensor_h)
            glob.append(zero_tensor_trans)
            glob_.append(zero_tensor_trans)

        gen_ims = []
        for t in range(in_len + out_len):  # loop every seqs
            if t < in_len:
                inputs = x[t]
            else:
                if mask is None:  # Test mode
                    inputs = x_gen
                else:  # Train mode using schedule sampling
                    inputs = mask[t - in_len] * x[t] + (1 - mask[t - in_len]) * x_gen

            inputs = self.embed(inputs)
            concat_cx = torch.cat([c[0], inputs], dim=1)
            glob[0] = concat_cx + self.TA(concat_cx)
            h[0], c[0], m = self.lstm[0](inputs, h[0], c[0], m)
            diff_l = h[0] - h[0]
            # loop subsequent layers
            for i in range(1, self.n_layers):
                if t == 0:
                    diff_t = glob_[i-1] - glob_[i-1]
                else:
                    diff_t = glob[i-1] - glob_[i-1]
                concat_cx = torch.cat([c[i], h[i - 1]], dim=1)
                glob[i] = concat_cx + self.TA(concat_cx)
                c[i], h[i], m, dt, dl = self.lstm[i](diff_t, diff_l, h[i - 1], c[i], h[i], m, dt, dl)
                diff_l = h[i] - h[i - 1]

            glob_ = glob # record current timestamp's glob
            x_gen = self.output(h[self.n_layers - 1])

            gen_ims.append(x_gen)

        prediction = torch.stack(gen_ims, dim=0)  # list to tensor
        # print(len(gen_ims), prediction.shape)
        # raise IOError
        return prediction

def get_HabLSTM(input_chans=1, output_chans=1, hidden_size=64, filter_size=5, num_layers=4, img_size=64):
    model = HabLSTM(input_chans, output_chans, hidden_size, filter_size, num_layers)
    return model

# if __name__ == "__main__":
#     device = "cuda"
#     model = HabLSTM(input_size=64, output_chans=64, hidden_size=64, filter_size=3, num_layers=4).to(device)
#     print_network(model, 'HabLSTM')

#     x_input = torch.ones(1, 64, 32, 32).to(device)

#     output = model(x_input)
#     print(output.shape)