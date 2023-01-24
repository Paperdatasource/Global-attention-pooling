class GAP(torch.nn.Module):
    def __init__(self,in_channels):
        """
        :param in_channels: The feature dimension of input node representation matrix.
        """
        super(GAP, self).__init__()
        self.gen_nn=nn.Linear(in_channels,int(in_channels),bias=False)
        self.other_nn=nn.Linear(in_channels,int(in_channels),bias=False)
        self.value_nn = nn.Linear(in_channels, int(in_channels),bias=False)
        self.bn=nn.BatchNorm1d(int(in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.gen_nn.reset_parameters()
        self.other_nn.reset_parameters()
        self.value_nn.reset_parameters()

    def forward(self, x, gen_index,batch,batchsize):
        """
        :param x: Node representation matrix, H_m
        :param gen_index: Generator index. e.g., [29,30,....,38] are generator index out of node index [0,...,38].
        :param batch: Link each node to each batch. e.g., [0,0,0,...1,1,1,....n,n,n]
        :param batchsize: Size of the present batch.
        :return:
        """

        x_g=self.gen_nn(x)

        new_x,mask_x,index_x=dense_batch23Dbatch(x,batch,batchsize)
        new_x_o = self.other_nn(new_x)
        new_x_v = self.value_nn(new_x)
        x_gen=x_g[gen_index,:]
        batch_gen=batch[gen_index]
        new_x_g,mask_x_g,index_g=dense_batch23Dbatch(x_gen,batch_gen,batchsize)
        x_g_o=torch.matmul(new_x_g,new_x_o.permute(0,2,1))/math.sqrt(new_x.size(2))
        # x_g_o = torch.matmul(new_x_g, new_x_o.permute(0, 2, 1))
        x_g_o=x_g_o.permute(0,2,1).reshape(x_g_o.size(0)*x_g_o.size(2),-1)
        mask_x_new=x_g_o.new_ones(x_g_o.size(0))
        mask_x_new[mask_x]=0
        x_g_o[mask_x_new.bool(),:]=torch.finfo(x_g_o.dtype).min
        x_g_o=x_g_o.reshape(batchsize,new_x_o.size(1),-1).permute(0,2,1)
        x_g_o=F.softmax(x_g_o,dim=2)
        final_x=torch.matmul(x_g_o,new_x_v)
        final_x=final_x.reshape(final_x.size(0)*final_x.size(1),-1)
        final_x=final_x[mask_x_g]
        _, new_index_g = torch.sort(index_g)
        final_x=final_x[new_index_g,:]
        final_x=F.relu(self.bn(final_x))
        return final_x
